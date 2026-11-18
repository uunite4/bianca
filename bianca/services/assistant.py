
import asyncio

from starlette.concurrency import run_in_threadpool

from . import wakeword, conversation
from ..logging_config import log_event

from ..config import FOLLOW_UP_TIMEOUT


async def assistant_loop(app, stop_event):
    """
    Main assistant loop:

    Wake word -> command -> response -> follow-up window
    -> repeat or return to wake-word listening.
    """

    model = app.state.wakeword_model
    whisper_model = app.state.whisper
    stop_whisper_model = app.state.stop_whisper
    tts_model = app.state.tts
    messages = app.state.conversation_messages

    log_event("assistant_started")

    print("\nBianca is ready. Say 'Bianca' to wake her.")
    print("Press Ctrl+C to stop.\n")

    while not stop_event.is_set():
        try:
            # --------------------------------
            # 1. WAIT FOR THE WAKE WORD
            # --------------------------------

            audio = await run_in_threadpool(
                wakeword.listen_for_command,
                model,
                stop_event,
            )

            if stop_event.is_set():
                break

            if audio.size == 0:
                continue

            log_event("wakeword_command_captured")

            # --------------------------------
            # 2. PROCESS COMMAND AND FOLLOW-UPS
            # --------------------------------

            while audio.size > 0 and not stop_event.is_set():

                async with app.state.conversation_lock:
                    result = await conversation.process_turn(
                        whisper_model,
                        tts_model,
                        messages,
                        audio,
                        stop_whisper_model,
                    )

                log_event(
                    "assistant_turn_finished",
                    status=result["status"],
                )

                if stop_event.is_set():
                    break

                # Don't enter follow-up mode if transcription
                # was empty or processing did not complete.
                if result["status"] != "completed":
                    break

                # --------------------------------
                # 3. WAIT FOR A FOLLOW-UP
                # --------------------------------

                audio = await run_in_threadpool(
                    wakeword.listen_for_follow_up,
                    stop_event,
                    FOLLOW_UP_TIMEOUT,
                )

                if stop_event.is_set():
                    break

                if audio.size > 0:
                    log_event("follow_up_captured")

        except asyncio.CancelledError:
            raise

        except Exception as error:
            log_event(
                "assistant_error",
                error=str(error),
            )

            print(f"Assistant error: {error}")

            if not stop_event.is_set():
                await asyncio.sleep(2)

    log_event("assistant_stopped")
    print("\nBianca has stopped.")
