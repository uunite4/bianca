import sounddevice as sd

MICROPHONE_ID = 9
SAMPLE_RATE = 48000

print(sd.query_devices())

print("\nStarting recording...")

audio = sd.rec(
    int(5 * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MICROPHONE_ID
)

sd.wait()

print("Recording successful!")