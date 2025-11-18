# agents/examples/filler_interrupt_handler.py

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Set

from livekit.agents import AgentSession, UserInputTranscribedEvent
from livekit.agents.voice import AgentStateChangedEvent, UserStateChangedEvent

logger = logging.getLogger("filler_interrupt_handler")


def _normalize_text(text: str) -> str:
    """
    Lowercase and remove extra punctuation/whitespace for comparison.
    """
    text = text.lower().strip()
    # remove simple punctuation, keep basic characters
    text = re.sub(r"[^\w\s']", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_words(text: str) -> list[str]:
    """
    Split normalized text into tokens.
    """
    if not text:
        return []
    return text.split()


@dataclass
class FillerAwareInterruptConfig:
    ignored_words: Set[str] = field(default_factory=set)
    interrupt_command_words: Set[str] = field(default_factory=set)
    min_confidence: float = 0.0  # optional, only used if confidence extractor is provided


class FillerAwareInterruptHandler:
    """
    Extension layer around a LiveKit AgentSession that:

    - Ignores configured filler words while the agent is SPEAKING.
    - Treats the same words as valid input when the agent is not speaking.
    - Still immediately interrupts when real commands like "stop", "wait" appear.
    - Does NOT modify base VAD, only uses events + explicit session.interrupt().

    Usage:
        handler = FillerAwareInterruptHandler(session)
        handler.start()
    """

    def __init__(
        self,
        session: AgentSession,
        config: Optional[FillerAwareInterruptConfig] = None,
        confidence_extractor: Optional[
            Callable[[UserInputTranscribedEvent], Optional[float]]
        ] = None,
    ) -> None:
        self._session = session
        self._config = config or self._load_config_from_env()
        self._confidence_extractor = confidence_extractor

        self._agent_state: str = "initializing"
        self._user_state: str = "listening"

        # Used to debounce / group short utterances if needed
        self._lock = asyncio.Lock()

        logger.info(
            "FillerAwareInterruptHandler initialized | ignored_words=%s | command_words=%s",
            sorted(self._config.ignored_words),
            sorted(self._config.interrupt_command_words),
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """
        Attach to the AgentSession events.
        Call this once after the session is created, before session.start().
        """
        self._session.on("agent_state_changed", self._on_agent_state_changed)
        self._session.on("user_state_changed", self._on_user_state_changed)
        self._session.on("user_input_transcribed", self._on_user_input_transcribed)

        logger.info("FillerAwareInterruptHandler event listeners registered.")

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def _on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        logger.debug("agent_state_changed: %s -> %s", ev.old_state, ev.new_state)
        self._agent_state = ev.new_state

    def _on_user_state_changed(self, ev: UserStateChangedEvent) -> None:
        logger.debug("user_state_changed: %s -> %s", ev.old_state, ev.new_state)
        self._user_state = ev.new_state

    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        """
        Main logic: called whenever the STT produces a transcript.

        We only care about:
        - final transcripts (ev.is_final == True)
        - when the agent is currently speaking
        """
        # Fire and forget async handler so we never block the event loop.
        asyncio.create_task(self._handle_transcript(ev), name="handle_transcript")

    async def _handle_transcript(self, ev: UserInputTranscribedEvent) -> None:
        async with self._lock:
            raw = ev.transcript or ""
            norm = _normalize_text(raw)
            tokens = _split_words(norm)

            if not ev.is_final:
                # partial transcripts: we just log and wait for final
                logger.debug(
                    "[TRANSCRIPT_PARTIAL] text='%s' | lang=%s | agent_state=%s | user_state=%s",
                    raw,
                    ev.language,
                    self._agent_state,
                    self._user_state,
                )
                return

            logger.info(
                "[TRANSCRIPT_FINAL] text='%s' | norm='%s' | lang=%s | agent_state=%s | user_state=%s",
                raw,
                norm,
                ev.language,
                self._agent_state,
                self._user_state,
            )

            # Optional: external confidence (depends on STT provider).
            confidence = None
            if self._confidence_extractor is not None:
                try:
                    confidence = self._confidence_extractor(ev)
                except Exception:
                    logger.exception("Error in confidence_extractor")
            if confidence is not None:
                logger.debug("ASR confidence: %.3f", confidence)

            # If agent is not speaking, we don't filter. Let it go as normal user input.
            if self._agent_state != "speaking":
                logger.debug(
                    "Agent is not speaking; no filler filtering applied for this transcript."
                )
                return

            # While agent IS speaking, decide if this is filler-only or real interruption.
            if self._is_command(tokens):
                logger.warning(
                    "[INTERRUPTION_VALID] Command detected while agent speaking. "
                    "text='%s'",
                    raw,
                )
                # Explicitly interrupt, even though allow_interruptions=False is set on speech.
                self._session.interrupt()
                return

            if self._is_filler_only(tokens, confidence):
                logger.info(
                    "[INTERRUPTION_IGNORED] Filler-only utterance while agent speaking. "
                    "text='%s'",
                    raw,
                )
                # Do nothing → agent continues speaking uninterrupted.
                return

            # Mixed filler + content: treat as valid interruption
            if tokens:
                logger.warning(
                    "[INTERRUPTION_VALID] Non-filler speech while agent speaking. "
                    "text='%s'",
                    raw,
                )
                self._session.interrupt()
            else:
                # No tokens at all: let built-in false interruption handling do its work.
                logger.info(
                    "[INTERRUPTION_FALLBACK] Empty transcript; rely on false_interruption_timeout."
                )

    # -------------------------------------------------------------------------
    # Helper logic
    # -------------------------------------------------------------------------

    def _is_command(self, tokens: Iterable[str]) -> bool:
        joined = " ".join(tokens)
        # Check both full phrase and per-token match
        if joined in self._config.interrupt_command_words:
            return True
        return any(tok in self._config.interrupt_command_words for tok in tokens)

    def _is_filler_only(
        self, tokens: Iterable[str], confidence: Optional[float]
    ) -> bool:
        tokens = list(tokens)
        if not tokens:
            return False  # let false interruption logic handle pure silence / noise

        # If we have a confidence extractor and it's VERY low, treat as ignorable background noise.
        if confidence is not None and confidence < self._config.min_confidence:
            logger.info(
                "Treating low-confidence transcript as filler/background. conf=%.3f",
                confidence,
            )
            return True

        # All tokens must be in ignored_words
        return all(tok in self._config.ignored_words for tok in tokens)

    # -------------------------------------------------------------------------
    # Config loading
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_word_list(value: str) -> Set[str]:
        if not value:
            return set()
        parts = [p.strip().lower() for p in value.split(",")]
        return {p for p in parts if p}

    @classmethod
    def _load_config_from_env(cls) -> FillerAwareInterruptConfig:
        ignored = cls._parse_word_list(
            os.getenv("IGNORED_FILLER_WORDS", "uh,umm,um,hmm,haan")
        )
        commands = cls._parse_word_list(
            os.getenv("INTERRUPT_COMMAND_WORDS", "stop,wait,no,not that,cancel")
        )
        try:
            min_conf = float(os.getenv("FILLER_MIN_CONFIDENCE", "0.0"))
        except ValueError:
            min_conf = 0.0

        return FillerAwareInterruptConfig(
            ignored_words=ignored,
            interrupt_command_words=commands,
            min_confidence=min_conf,
        )
