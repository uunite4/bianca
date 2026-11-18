from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Ollama
OLLAMA_URL_BASE = "http://127.0.0.1:11434"
OLLAMA_URL_CHAT = "/api/chat"
LLM_MODEL = "bianca_v0.2"

# Kokoro (TTS)
KOKORO_MODEL_PATH = BASE_DIR / "models" / "kokoro" / "kokoro-v1.0.int8.onnx"
KOKORO_VOICES_PATH = BASE_DIR / "models" / "kokoro" / "voices-v1.0.bin"
TTS_VOICE = "af_bella"
TTS_SPEED = 1.2
TTS_LANGUAGE = "en-us"
SENTENCE_PAUSE_SECONDS = 0.2
CLAUSE_PAUSE_SECONDS = 0.1    # after a chunk ending in , ; :
TTS_SAMPLE_RATE = 24000

# TTS speed once the LLM has finished writing (seconds of synthesis per
# second of audio). Your logs show about 0.65.
TTS_IDLE_RATIO = 0.5

# Start-gate: hold the first word back until the audio buffered so far
# can cover the synthesis of the upcoming chunk(s).
GATE_LOOKAHEAD_CHUNKS = 1
STREAM_OPEN_SECONDS = 0
START_MARGIN_SECONDS = -1.0
MAX_START_WAIT_SECONDS = 1.00

# Whisper (STT)
WHISPER_MODEL_SIZE = "base.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# WAKWORD
WAKEWORD_MODEL_PATH = BASE_DIR / "models" / "wakeword" / "bianca.onnx"

# Microphone
MICROPHONE_ID = 1
SAMPLE_RATE = 48000
RECORDING_DURATION = 5

# Conversation settings
FOLLOW_UP_TIMEOUT = 8.0

# CPU threads. Your CPU: 4 physical cores / 8 logical.
# None = use the library default.
LLM_NUM_THREAD = None
TTS_NUM_THREADS = None