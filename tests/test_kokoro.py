from time import perf_counter
from kokoro_onnx import Kokoro
import soundfile as sf

MODEL_PATH = "models/kokoro/kokoro-v1.0.onnx"
VOICES_PATH = "models/kokoro/voices-v1.0.bin"

start = perf_counter()
tts = Kokoro(MODEL_PATH, VOICES_PATH)
print(f"Model loading: {perf_counter() - start:.2f}s")

text = "Sure sir. I wrote in your notes: 'Project X: add a pool table'"

start = perf_counter()
audio, sample_rate = tts.create(
    text,
    voice="af_bella",
    speed=1.0,
    lang="en-us"
)
print(f"Speech generation: {perf_counter() - start:.2f}s")

sf.write("bella_test.wav", audio, sample_rate, subtype="PCM_16")
print(f"Audio duration: {len(audio) / sample_rate:.2f}s")

tts.voices.close()