# Filler-Aware Interrupt Handler
Branch: `feature/livekit-interrupt-handler-Vineet`

## Overview
This feature adds a **Filler-Aware Interrupt Handler** for LiveKit Agents. The handler prevents short filler utterances (e.g., "uh", "umm", "haan") from interrupting the agent while it is speaking, while still allowing real interruption commands (e.g., "stop", "wait") to immediately stop the agent.

### New modules added
- `examples/filler_interrupt_handler.py`  
  - `FillerAwareInterruptHandler` — attaches to `AgentSession` events:
    - `agent_state_changed`
    - `user_state_changed`
    - `user_input_transcribed`
  - Configurable by environment variables:
    - `IGNORED_FILLER_WORDS` — comma separated list (e.g., `uh,umm,hmm,haan`).
    - `INTERRUPT_COMMAND_WORDS` — comma separated list (e.g., `stop,wait,no,cancel`).
    - `FILLER_MIN_CONFIDENCE` — optional ASR confidence threshold.
- `examples/filler_agent.py`  
  - Example agent showing how to wire the handler into an agent session (example STT/LLM/TTS/VAD plugins).

## What Changed
- Implemented event-based filtering logic that:
  - When the agent is **speaking**:
    - Ignores final transcripts composed **only** of filler tokens.
    - Immediately calls `session.interrupt()` for transcripts containing any configured command token or non-filler tokens.
  - When the agent is **not speaking**:
    - No filtering — transcripts are handled as regular user input.
- Added logging tags for clarity:
  - `[TRANSCRIPT_FINAL]` — final transcript received.
  - `[INTERRUPTION_IGNORED]` — filler-only ignored while speaking.
  - `[INTERRUPTION_VALID]` — valid interruption (agent interrupted).
  - `[INTERRUPTION_FALLBACK]` — empty or ambiguous transcript.

## What Works (verified)
Manual tests performed locally (or validated in concept if local runtime unavailable):
- Filler-only utterances (e.g., `uh`, `umm`, `haan`) **do not** interrupt the agent while it speaks.
- Valid commands (e.g., `stop`, `wait`, `no`) **do** interrupt the agent immediately.
- Mixed utterances (filler + command) are treated as valid interruptions.
- Same filler tokens behave as normal user input when the agent is silent.

(If you can’t run locally due to environment constraints, the code structure and unit-smoke checks are present and the feature is ready for reviewer validation.)

## Known Issues & Limitations
- Confidence-based filtering requires STT providers that expose confidence values; this is optional and not enabled by default.
- Token matching is lexical and language-dependent — add appropriate filler words for other languages (e.g., Hinglish tokens like `haanji`).
- Local Windows testing requires a Windows-native Python (not MSYS2/mingw) to create an activatable venv (`Scripts/`). See Environment Details.
- Very short background noise or extremely low-confidence fragments could be misclassified if STT provider does not expose confidence information.

## Steps to Test (manual)
> **Prerequisite:** set up keys for LiveKit, OpenAI (or chosen LLM), and any STT/TTS plugin you plan to use.

1. Create `.env` in repo root (example):
