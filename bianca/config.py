from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Ollama
OLLAMA_URL_BASE = "http://127.0.0.1:11434"
OLLAMA_URL_CHAT = "/api/chat"
LLM_MODEL = "bianca_v0.2"

# Kokoro (TTS)
KOKORO_MODEL_PATH = BASE_DIR / "models" / "kokoro" / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = BASE_DIR / "models" / "kokoro" / "voices-v1.0.bin"
TTS_VOICE = "af_bella"
TTS_SPEED = 1.2
TTS_LANGUAGE = "en-us"

# Whisper (STT)
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# WAKWORD
WAKEWORD_MODEL_PATH = BASE_DIR / "models" / "wakeword" / "bianca.onnx"

# Microphone
MICROPHONE_ID = 1
SAMPLE_RATE = 48000
RECORDING_DURATION = 5

# Conversation settings
FOLLOW_UP_TIMEOUT = 10.0