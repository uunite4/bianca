from time import perf_counter
from io import BytesIO

import asyncio
import sounddevice as sd
import soundfile as sf
import re

from starlette.concurrency import run_in_threadpool

from . import llm, stt
from . import tts as tts_service

from ..logging_config import log_event

def is_end_conversation_phrase(text):
    """Check whether the user wants to end the conversation."""

    normalized = re.sub(
        r"[^\w\s]",
        "",
        text.lower(),
    )

    normalized = " ".join(normalized.split())

    return normalized in {
        "thanks bianca",
        "thank you bianca",
        "thanks"
    }

def clean_transcription(text: str) -> str:
    """
    Remove Bianca's wake word if it appears at the
    beginning of the transcription.
    """

    return re.sub(
        r"^\s*bianca\b[\s,.:;!?-]*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()

async def process_turn(
    whisper_model,
    tts_model,
    messages: list[dict],
    audio,
):
    request_started = perf_counter()

    # ------------------------------
    # 1. LISTEN AND TRANSCRIBE
    # ------------------------------

    stt_started = perf_counter()

    transcription = await run_in_threadpool(
        stt.transcribe_audio,
        whisper_model,
        audio,
    )

    raw_text = transcription["text"]
    language = transcription["language"]

    # Check for an end-of-conversation phrase.
    end_conversation = is_end_conversation_phrase(raw_text)

    user_text = clean_transcription(raw_text)

    if not user_text:
        log_event(
            "conversation_empty",
            stt_duration_ms=round(
                (perf_counter() - stt_started) * 1000,
                2
            )
        )

        return {
            "status": "empty",
            "text": "",
            "reply": None,
        }

    log_event(
        "conversation_transcribed",
        text=user_text,
        language=language,
        stt_duration_ms=round(
            (perf_counter() - stt_started) * 1000,
            2
        )
    )

    # ------------------------------
    # 2. GENERATE RESPONSE
    # ------------------------------

    conversation_messages = [
        *messages,
        {
            "role": "user",
            "content": user_text,
        }
    ]

    llm_result = await llm.generate(
        conversation_messages
    )

    reply = llm_result["reply"]

    # ------------------------------
    # 3. GENERATE SPEECH
    # ------------------------------

    tts_started = perf_counter()

    audio_bytes, tts_metrics = await run_in_threadpool(
        tts_service.generate,
        tts_model,
        reply,
        llm_result["request_id"]
    )

    log_event(
        "conversation_tts_completed",
        tts_duration_ms=round(
            (perf_counter() - tts_started) * 1000,
            2
        )
    )

    # ------------------------------
    # 4. PLAY AUDIO
    # ------------------------------

    playback_started = perf_counter()

    audio, sample_rate = sf.read(
        BytesIO(audio_bytes),
        dtype="float32"
    )

    await run_in_threadpool(
        play_audio,
        audio,
        sample_rate
    )

    log_event(
        "conversation_playback_completed",
        playback_duration_ms=round(
            (perf_counter() - playback_started) * 1000,
            2
        )
    )

    # ------------------------------
    # 5. UPDATE HISTORY
    # ------------------------------

    messages.append({
        "role": "user",
        "content": user_text,
    })

    messages.append({
        "role": "assistant",
        "content": reply,
    })

    total_duration_ms = round(
        (perf_counter() - request_started) * 1000,
        2
    )

    log_event(
        "conversation_turn_completed",
        total_duration_ms=total_duration_ms
    )

    return {
        "status": "end-conversation" if end_conversation else "completed",
        "text": user_text,
        "language": language,
        "reply": reply,
        "request_id": llm_result["request_id"],
        "total_duration_ms": total_duration_ms,
    }


def play_audio(audio, sample_rate):

    sd.play(
        audio,
        samplerate=sample_rate
    )

    sd.wait()
