# agents/examples/filler_agent.py

import asyncio
import logging
import os

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai, cartesia, silero, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from .filler_interrupt_handler import FillerAwareInterruptHandler

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("filler_agent")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful, friendly voice AI assistant. "
                "Speak concisely and naturally, and let the user interrupt you "
                "with commands like 'stop' or 'wait'."
            )
        )


async def entrypoint(ctx: agents.JobContext) -> None:
    """
    Main entrypoint used by livekit-agents CLI.
    """
    await ctx.connect()  # connect to LiveKit room

    # --- Configure STT, LLM, TTS, VAD, Turn detection ---

    # Example: Deepgram STT, OpenAI LLM, Cartesia TTS, Silero VAD
    stt = deepgram.STT(model="nova-3", language="en")  # or "multi" for multi-language
    llm = openai.LLM(model="gpt-4o-mini")  # adjust model as you like
    tts = cartesia.TTS()  # default Cartesia voice, configurable by env/API
    vad = silero.VAD.load()
    turn_detection = MultilingualModel()

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        turn_detection=turn_detection,
        # Turn/interrupt tuning (can tweak if needed)
        allow_interruptions=True,  # still allow internal interruptions; we layer on top
        # false_interruption_timeout and resume_false_interruption are left default,
        # which already resume when VAD fires but no words are recognized.
        # :contentReference[oaicite:11]{index=11}
    )

    # Attach our filler-aware handler BEFORE starting the session
    handler = FillerAwareInterruptHandler(session)
    handler.start()

    logger.info("Starting AgentSession with filler-aware interrupt handler...")

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        input=RoomInputOptions(
            # We want the agent to speak AND listen in the room.
            # See docs for details, default is usually fine. :contentReference[oaicite:12]{index=12}
        ),
    )

    # Optionally greet the user with initial speech.
    # IMPORTANT: we set allow_interruptions=False here and let our handler
    # decide when to call session.interrupt() based on transcripts.
    greet_handle = await session.say(
        "Hi! I'm your assistant. Feel free to stop me by saying 'stop' or 'wait'.",
        allow_interruptions=False,
    )
    await greet_handle.wait_for_playout()

    # Main loop: typical pattern is just to wait until the session closes.
    # LiveKit Agents manages turn detection, STT, LLM, TTS in the background.
    await session.wait_until_closed()
    logger.info("AgentSession closed.")


if __name__ == "__main__":
    # Run with:
    #   python -m livekit.agents.cli run agents.examples.filler_agent:entrypoint dev
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # optional: set worker name, etc.
        )
    )
