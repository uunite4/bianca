import sounddevice as sd
import soundfile as sf

MICROPHONE_ID = 9
DURATION = 5
SAMPLE_RATE = 48000

print("Recording for 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MICROPHONE_ID
)

sd.wait()

sf.write("test_recording.wav", audio, SAMPLE_RATE)

print("Recording saved!")

print("Playing recording...")

recording, sample_rate = sf.read("test_recording.wav")

sd.play(
    recording,
    samplerate=sample_rate,
    device=8
)

sd.wait()

print("Playback finished!")