from time import perf_counter
from io import BytesIO
import numpy as np

import soundfile as sf

from ..config import (
    TTS_VOICE,
    TTS_SPEED,
    TTS_LANGUAGE,
    SENTENCE_PAUSE_SECONDS,
    CLAUSE_PAUSE_SECONDS,
)
from ..logging_config import log_event


def generate(tts_model, text: str, request_id: str):

    request_started = perf_counter()

    # Generate speech using Kokoro
    generation_started = perf_counter()

    audio, sample_rate = tts_model.create(
        text,
        voice=TTS_VOICE,
        speed=TTS_SPEED,
        lang=TTS_LANGUAGE
    )

    generation_duration_ms = round(
        (perf_counter() - generation_started) * 1000, 2
    )

    # Calculate the duration of the generated audio
    audio_duration_seconds = len(audio) / sample_rate

    # Encode the audio as WAV
    encoding_started = perf_counter()

    audio_buffer = BytesIO()

    sf.write(
        audio_buffer,
        audio,
        sample_rate,
        format="WAV",
        subtype="PCM_16"
    )

    encoding_duration_ms = round(
        (perf_counter() - encoding_started) * 1000, 2
    )

    audio_bytes = audio_buffer.getvalue()

    total_duration_ms = round(
        (perf_counter() - request_started) * 1000, 2
    )

    metrics = {
        "generation_duration_ms": generation_duration_ms,
        "encoding_duration_ms": encoding_duration_ms,
        "total_duration_ms": total_duration_ms,
        "audio_duration_seconds": round(audio_duration_seconds, 2),
        "audio_size_bytes": len(audio_bytes),
        "sample_rate": sample_rate,
        "input_characters": len(text),
    }

    log_event(
        "speech_completed",
        request_id=request_id,
        **metrics
    )

    return audio_bytes, metrics

def synthesize(tts_model, text: str):
    """
    Generate raw audio for one chunk of text.
    Returns (float32 samples, sample_rate).
    """

    # Look at how the chunk ends (ignoring closing quotes/brackets).
    last = text.rstrip().rstrip("\"'”’)]")[-1:]

    if last in {".", "!", "?"}:
        pause_seconds = SENTENCE_PAUSE_SECONDS
        spoken = text
    elif last in {",", ";", ":"}:
        pause_seconds = CLAUSE_PAUSE_SECONDS
        spoken = text
    else:
        # No punctuation at the end (a list item, a heading, the last
        # words of a reply): treat it as a finished thought.
        pause_seconds = SENTENCE_PAUSE_SECONDS
        spoken = f"{text}."

    audio, sample_rate = tts_model.create(
        spoken,
        voice=TTS_VOICE,
        speed=TTS_SPEED,
        lang=TTS_LANGUAGE,
    )

    pause = np.zeros(int(sample_rate * pause_seconds), dtype=np.float32)

    return np.concatenate([audio.astype(np.float32), pause]), sample_rate