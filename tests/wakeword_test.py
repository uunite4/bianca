from pathlib import Path
import sys
import time

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from openwakeword.model import Model

from core.config import MICROPHONE_ID


# Model location
MODEL_PATH = PROJECT_ROOT / "models" / "wakeword" / "bianca.onnx"

# Microphone settings
INPUT_SAMPLE_RATE = 48000
WAKEWORD_SAMPLE_RATE = 16000

# openWakeWord works with 80 ms frames at 16 kHz.
FRAME_SIZE = 1280

# Corresponding frame size at 48 kHz.
INPUT_FRAME_SIZE = 3840

# Initial detection threshold.
THRESHOLD = 0.5


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Wake-word model not found: {MODEL_PATH.resolve()}"
        )

    print("Loading Bianca's wake-word model...")

    model = Model(
        wakeword_models=[str(MODEL_PATH)],
        inference_framework="onnx",
    )

    print("Loaded models:", list(model.models.keys()))
    print("Starting microphone...")
    print('Say "Bianca" to test. Press Ctrl+C to stop.')

    last_print = 0

    with sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=INPUT_FRAME_SIZE,
        device=MICROPHONE_ID,
    ) as stream:

        while True:
            audio, overflowed = stream.read(INPUT_FRAME_SIZE)

            if overflowed:
                print("Warning: microphone overflow")

            # Convert mono float32 audio to 16 kHz.
            audio = audio[:, 0]
            audio = resample_poly(audio, up=1, down=3)

            # Convert float32 [-1, 1] to int16 PCM.
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)

            # Run wake-word detection.
            predictions = model.predict(audio)

            now = time.time()

            # Print scores twice per second.
            if now - last_print >= 0.5:
                scores = {
                    name: round(float(score), 3)
                    for name, score in predictions.items()
                }
                print("Scores:", scores)
                last_print = now

            # Report a detection.
            for name, score in predictions.items():
                if float(score) >= THRESHOLD:
                    print(f"\n*** WAKE WORD DETECTED: {name} ({score:.3f}) ***\n")
                    model.reset()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nWake-word test stopped.")