
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from openwakeword.model import Model

from ..config import MICROPHONE_ID, WAKEWORD_MODEL_PATH as MODEL_PATH
from ..logging_config import log_event

INPUT_SAMPLE_RATE = 48000
WAKEWORD_SAMPLE_RATE = 16000

FRAME_DURATION = 0.08
INPUT_FRAME_SIZE = 3840

WAKEWORD_THRESHOLD = 0.5

SILENCE_THRESHOLD = 0.010
SILENCE_DURATION = 0.8

PRE_ROLL_DURATION = 1.6
FOLLOW_UP_PRE_ROLL_DURATION = 0.4

MAX_COMMAND_DURATION = 20.0

# Follow-up speech detection
MIN_SPEECH_THRESHOLD = 0.006
NOISE_MULTIPLIER = 2.5
SPEECH_FRAMES_REQUIRED = 2


def load_model():
    """Load Bianca's custom wake-word model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Wake-word model not found: {MODEL_PATH}"
        )

    return Model(
        wakeword_models=[str(MODEL_PATH)],
        inference_framework="onnx",
    )


def _read_frame(stream, stop_event):
    """Read and resample one 80 ms microphone frame."""

    if stop_event is not None and stop_event.is_set():
        return None

    frame, overflowed = stream.read(INPUT_FRAME_SIZE)

    if stop_event is not None and stop_event.is_set():
        return None

    if overflowed:
        print("Warning: microphone input overflowed")

    frame = frame[:, 0]

    return resample_poly(
        frame,
        up=1,
        down=3,
    ).astype(np.float32)


def _get_volume(audio):
    """Calculate the RMS volume of an audio frame."""

    return float(np.sqrt(np.mean(audio ** 2)))


def _capture_until_silence(
    stream,
    initial_frames,
    stop_event,
    silence_threshold,
):
    """Record speech until silence or the maximum duration."""

    command_frames = list(initial_frames)

    silence_frames_required = int(
        SILENCE_DURATION / FRAME_DURATION
    )

    silence_frames = 0
    recording_started = perf_counter()
    reason = "unknown"

    while True:
        frame = _read_frame(stream, stop_event)

        if frame is None:
            return np.array([], dtype=np.float32)

        command_frames.append(frame.copy())

        volume = _get_volume(frame)

        if volume < silence_threshold:
            silence_frames += 1
        else:
            silence_frames = 0

        elapsed = perf_counter() - recording_started

        if silence_frames >= silence_frames_required:
            print("Speech finished.")
            reason = "silence"
            break

        if elapsed >= MAX_COMMAND_DURATION:
            print("Maximum command duration reached.")
            reason = "max_duration"
            break

    audio = np.concatenate(command_frames)

    log_event(
        "capture_finished",
        reason=reason,
        audio_seconds=round(len(audio) / WAKEWORD_SAMPLE_RATE, 2),
        silence_threshold=round(float(silence_threshold), 4),
    )

    return audio


def listen_for_command(model, stop_event=None):
    """
    Wait for 'Bianca', then capture the command.

    Returns mono float32 audio sampled at 16 kHz.
    """

    pre_roll_frames = int(
        PRE_ROLL_DURATION / FRAME_DURATION
    )

    audio_buffer = deque(maxlen=pre_roll_frames)

    print("Waiting for Bianca...")

    with sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=INPUT_FRAME_SIZE,
        device=MICROPHONE_ID,
    ) as stream:

        # Phase 1: Wait for the wake word.
        while True:
            audio_16k = _read_frame(stream, stop_event)

            if audio_16k is None:
                return np.array([], dtype=np.float32)

            audio_buffer.append(audio_16k.copy())

            audio_int16 = np.clip(
                audio_16k * 32767,
                -32768,
                32767,
            ).astype(np.int16)

            predictions = model.predict(audio_int16)

            detected = any(
                float(score) >= WAKEWORD_THRESHOLD
                for score in predictions.values()
            )

            if detected:
                print("Bianca detected!")

                initial_frames = list(audio_buffer)

                model.reset()
                break

        # Phase 2: Capture the command.
        print("Recording command...")

        return _capture_until_silence(
            stream,
            initial_frames,
            stop_event,
            SILENCE_THRESHOLD,
        )


def listen_for_follow_up(stop_event=None, timeout=10.0):
    """
    Wait for the user to start speaking without requiring
    the wake word.

    Returns empty audio if the timeout expires.
    """

    pre_roll_frames = int(
        FOLLOW_UP_PRE_ROLL_DURATION / FRAME_DURATION
    )

    audio_buffer = deque(maxlen=pre_roll_frames)

    print(
        f"Listening for a follow-up ({timeout:.0f}s)..."
    )

    with sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=INPUT_FRAME_SIZE,
        device=MICROPHONE_ID,
    ) as stream:

        waiting_started = perf_counter()

        noise_floor = None
        speech_frames = 0

        # Phase 1: Wait for the user to start speaking.
        while True:
            if stop_event is not None and stop_event.is_set():
                return np.array([], dtype=np.float32)

            if perf_counter() - waiting_started >= timeout:
                print("Follow-up window expired.")
                return np.array([], dtype=np.float32)

            audio_16k = _read_frame(stream, stop_event)

            if audio_16k is None:
                return np.array([], dtype=np.float32)

            volume = _get_volume(audio_16k)

            audio_buffer.append(audio_16k.copy())

            if noise_floor is None:
                noise_floor = volume
            else:
                threshold = max(
                    MIN_SPEECH_THRESHOLD,
                    noise_floor * NOISE_MULTIPLIER,
                )

                # Update the estimated background noise only
                # while the volume is below the speech threshold.
                if volume < threshold:
                    noise_floor = (
                        noise_floor * 0.95
                        + volume * 0.05
                    )

            threshold = max(
                MIN_SPEECH_THRESHOLD,
                noise_floor * NOISE_MULTIPLIER,
            )

            if volume >= threshold:
                speech_frames += 1
            else:
                speech_frames = 0

            if speech_frames >= SPEECH_FRAMES_REQUIRED:
                print("Follow-up detected!")

                initial_frames = list(audio_buffer)
                break

        # Phase 2: Record the follow-up.
        print("Recording follow-up...")

        return _capture_until_silence(
            stream,
            initial_frames,
            stop_event,
            threshold,
        )
