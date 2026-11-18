import faulthandler
faulthandler.enable()
import asyncio
import signal
import threading
import onnxruntime as ort
from time import perf_counter
from types import SimpleNamespace

from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel

from bianca.config import (
    KOKORO_MODEL_PATH,
    KOKORO_VOICES_PATH,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    TTS_NUM_THREADS,
    STOP_WHISPER_MODEL,
    STOP_WHISPER_THREADS,
)

from bianca.services import llm
from bianca.services import tts as tts_service

from bianca.logging_config import logger, log_event
from bianca.services.wakeword import load_model
from bianca.services.assistant import assistant_loop

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def load_tts():
    """Load Kokoro, optionally with a fixed number of CPU threads."""

    if TTS_NUM_THREADS is None:
        return Kokoro(
            str(KOKORO_MODEL_PATH),
            str(KOKORO_VOICES_PATH),
        )

    options = ort.SessionOptions()
    options.intra_op_num_threads = TTS_NUM_THREADS

    session = ort.InferenceSession(
        str(KOKORO_MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    return Kokoro.from_session(
        session,
        str(KOKORO_VOICES_PATH),
    )

async def main():
    # Shared application state used by the assistant services.
    app = SimpleNamespace(
        state=SimpleNamespace(
            tts=None,
            whisper=None,
            stop_whisper=None,
            wakeword_model=None,
            conversation_messages=[],
            conversation_lock=asyncio.Lock(),
        )
    )

    # A thread-safe signal used to stop microphone listening.
    stop_event = threading.Event()

    def request_shutdown(signum, frame):
        logger.info("Shutdown requested.")
        stop_event.set()

    # Handle Ctrl+C.
    signal.signal(signal.SIGINT, request_shutdown)

    # Also handle termination signals where supported.
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)

    try:
        # 1. Load Kokoro.
        started = perf_counter()
        logger.info("Loading Bianca's voice...")

        app.state.tts = load_tts()

        log_event(
            "tts_loaded",
            startup_duration_ms=round(
                (perf_counter() - started) * 1000, 2
            ),
        )

        # 2. Load Whisper.
        started = perf_counter()
        logger.info("Loading Whisper...")

        app.state.whisper = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )

        log_event(
            "stt_loaded",
            startup_duration_ms=round(
                (perf_counter() - started) * 1000, 2
            ),
        )

        # 2b. Load a small, fast Whisper just for the stop phrases.
        if STOP_WHISPER_MODEL:
            started = perf_counter()
            logger.info("Loading the stop-phrase listener...")

            app.state.stop_whisper = WhisperModel(
                STOP_WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=STOP_WHISPER_THREADS,
            )

            log_event(
                "stop_whisper_loaded",
                startup_duration_ms=round(
                    (perf_counter() - started) * 1000, 2
                ),
            )

        # 3. Load the custom wake-word model.
        started = perf_counter()
        logger.info("Loading Bianca's wake-word model...")

        app.state.wakeword_model = load_model()

        log_event(
            "wakeword_loaded",
            startup_duration_ms=round(
                (perf_counter() - started) * 1000, 2
            ),
        )

        if stop_event.is_set():
            return

        # 4. Warm up, so the first real question isn't slow.
        started = perf_counter()
        logger.info("Warming up...")

        await llm.warm_up()
        tts_service.synthesize(app.state.tts, "Hello there.")

        log_event(
            "warmup_done",
            startup_duration_ms=round(
                (perf_counter() - started) * 1000, 2
            ),
        )

        # 5. Start the assistant.
        await assistant_loop(app, stop_event)

    finally:
        # Ensure the microphone listener receives the stop signal.
        stop_event.set()

        # Release Kokoro resources.
        if app.state.tts is not None:
            app.state.tts.voices.close()
            app.state.tts = None
            logger.info("Bianca's voice has been unloaded.")

        # Release references to the other models.
        app.state.whisper = None
        app.state.stop_whisper = None
        app.state.wakeword_model = None

        log_event("bianca_shutdown_complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Fallback in case the process receives an interrupt
        # before the custom signal handler is installed.
        print("\nBianca stopped.")
