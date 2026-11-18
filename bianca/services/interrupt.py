import re
from collections import deque
from time import perf_counter

import numpy as np
import sounddevice as sd

from ..config import (
    MICROPHONE_ID,
    STOP_CALIBRATION_SECONDS,
    STOP_LEVEL_MULTIPLIER,
    STOP_MIN_LEVEL,
    STOP_MAX_LEVEL,
    STOP_SPEECH_FRAMES,
    STOP_MIN_CAPTURE_SECONDS,
    STOP_MAX_CAPTURE_SECONDS,
    STOP_COOLDOWN_SECONDS,
    STOP_DEBUG_LEVELS,
)
from ..logging_config import log_event
from .wakeword import (
    INPUT_SAMPLE_RATE,
    INPUT_FRAME_SIZE,
    FRAME_DURATION,
    _read_frame,
    _get_volume,
)

# Whole-word phrases (lowercase, no punctuation). "stop" also covers
# "stop explaining", "stop talking", "please stop", "Bianca, stop"...
STOP_PHRASES = [
    "stop",
    "shut up",
    "be quiet",
    "enough",
    "thats enough",
    "that is enough",
]

# Quiet frames in a row that mean "the user finished speaking".
END_FRAMES = 4

def _words(text: str) -> str:
    """Lowercase, no punctuation, single spaces."""

    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def find_stop_phrase(text: str, spoken_text: str = ""):
    """
    Return the stop phrase found in `text`, or None.
    Phrases that Bianca is saying herself (`spoken_text`) don't count.
    """

    heard = f" {_words(text)} "
    own = f" {_words(spoken_text)} "

    for phrase in sorted(STOP_PHRASES, key=len, reverse=True):
        needle = f" {phrase} "

        if needle in heard and needle not in own:
            return phrase

    return None

def _is_own_speech(heard: str, spoken_text: str) -> bool:
    """True if most of the words heard are ones Bianca herself is saying."""

    words = _words(heard).split()

    if not words:
        return False

    own = set(_words(spoken_text).split())

    return sum(word in own for word in words) / len(words) >= 0.6

def _transcribe(whisper_model, audio) -> str:
    segments, _info = whisper_model.transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
    )

    return " ".join(segment.text.strip() for segment in segments).strip()


def listen_for_stop(whisper_model, cancel_event, get_spoken_text):
    """
    Listen to the microphone while Bianca is talking.

    Returns True as soon as a stop phrase is heard, or False when
    cancel_event is set (she finished talking).
    """

    pre_roll = deque(maxlen=int(0.4 / FRAME_DURATION))
    calibration_frames = int(STOP_CALIBRATION_SECONDS / FRAME_DURATION)
    min_capture_frames = int(STOP_MIN_CAPTURE_SECONDS / FRAME_DURATION)
    max_capture_frames = int(STOP_MAX_CAPTURE_SECONDS / FRAME_DURATION)

    with sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=INPUT_FRAME_SIZE,
        device=MICROPHONE_ID,
    ) as stream:

        echo_peak = 0.0        # how loud Bianca herself sounds in the mic
        frames_seen = 0
        loud_frames = 0
        cooldown_until = 0.0
        window_max = 0.0

        while True:
            frame = _read_frame(stream, cancel_event)

            if frame is None:
                return False

            volume = _get_volume(frame)
            pre_roll.append(frame.copy())
            frames_seen += 1

            # Phase 1: learn how loud she is.
            if frames_seen <= calibration_frames:
                echo_peak = min(max(echo_peak, volume), STOP_MAX_LEVEL)

                if frames_seen == calibration_frames:
                    log_event(
                        "stop_listener_calibrated",
                        echo_peak=round(echo_peak, 4),
                    )

                continue

            threshold = min(
                STOP_MAX_LEVEL,
                max(STOP_MIN_LEVEL, echo_peak * STOP_LEVEL_MULTIPLIER),
            )

            window_max = max(window_max, volume)

            if STOP_DEBUG_LEVELS and frames_seen % 25 == 0:
                log_event(
                    "stop_listener_levels",
                    echo_peak=round(echo_peak, 4),
                    threshold=round(threshold, 4),
                    recent_max=round(window_max, 4),
                )
                window_max = 0.0

            # Phase 2: wait for something clearly louder than her.
            if volume < threshold or perf_counter() < cooldown_until:
                loud_frames = 0

                if volume < threshold:
                    # Follow her loudness, forgetting slowly (about 11 s).
                    echo_peak = max(volume, echo_peak * 0.995)

                continue

            loud_frames += 1

            if loud_frames < STOP_SPEECH_FRAMES:
                continue

            log_event(
                "stop_listener_triggered",
                level=round(volume, 4),
                threshold=round(threshold, 4),
            )

            # Phase 3: record the whole phrase (at least a minimum length,
            # so a pause after "Bianca," doesn't cut it off).
            captured = list(pre_roll)
            peak_volume = volume
            quiet_frames = 0

            for i in range(max_capture_frames):
                frame = _read_frame(stream, cancel_event)

                if frame is None:
                    return False

                captured.append(frame.copy())
                level = _get_volume(frame)
                peak_volume = max(peak_volume, level)

                quiet_frames = quiet_frames + 1 if level < threshold else 0

                if i >= min_capture_frames and quiet_frames >= END_FRAMES:
                    break

            audio = np.concatenate(captured)

            whisper_started = perf_counter()
            heard = _transcribe(whisper_model, audio)
            whisper_ms = round((perf_counter() - whisper_started) * 1000, 2)

            spoken = get_spoken_text()
            phrase = find_stop_phrase(heard, spoken)

            log_event(
                "stop_listener_check",
                text=heard,
                matched=phrase,
                level=round(peak_volume, 4),
                threshold=round(threshold, 4),
                clip_s=round(len(audio) / 16000, 2),
                whisper_ms=whisper_ms,
            )

            if phrase:
                return True

            # No stop phrase. Only raise the bar if this was her own voice
            # (or just noise). If it was you, leave it, so your next
            # attempt isn't harder than this one.
            if _is_own_speech(heard, spoken):
                echo_peak = min(max(echo_peak, peak_volume), STOP_MAX_LEVEL)
            elif not heard.strip():
                echo_peak = min(
                    max(echo_peak, 0.8 * peak_volume),
                    STOP_MAX_LEVEL,
                )

            cooldown_until = perf_counter() + STOP_COOLDOWN_SECONDS
            loud_frames = 0
            pre_roll.clear()

            # Drop the audio that piled up while Whisper was running.
            available = stream.read_available

            if available:
                stream.read(available)