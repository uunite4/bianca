from faster_whisper import WhisperModel

MODEL_SIZE = "base"

print("loading whisper...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

print("whisper loaded!")

segments, info = model.transcribe(
    "test_recording.wav",
    beam_size=5
)

print(f"Detected language: {info.language}")

print("\nTranscription:")

for segment in segments:
    print(segment.text)