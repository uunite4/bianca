
import sounddevice as sd
import scipy.signal as signal

from ..config import (
    MICROPHONE_ID,
    SAMPLE_RATE,
    RECORDING_DURATION,
)

from ..logging_config import log_event


def transcribe(whisper_model):

    # Record audio from the microphone
    audio = sd.rec(
        int(RECORDING_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MICROPHONE_ID
    )

    sd.wait()

    # Resample from 48 kHz to 16 kHz
    audio_16k = signal.resample_poly(
        audio.flatten(),
        1,
        3
    )

    # Transcribe the audio
    segments, info = whisper_model.transcribe(
        audio_16k,
        language="en",
        beam_size=5,
        vad_filter=True
    )

    text = "".join(segment.text for segment in segments).strip()

    log_event(
        "stt_completed",
        text=text,
        language=info.language
    )

    return {
        "text": text,
        "language": info.language
    }


def transcribe_audio(whisper_model, audio):
    """
    Transcribe an already-recorded 16 kHz mono audio array.
    """

    segments, info = whisper_model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    log_event(
        "transcription",
        text=text,
        language=info.language,
    )

    return {
        "text": text,
        "language": info.language,
    }