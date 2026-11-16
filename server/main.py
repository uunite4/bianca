
from time import perf_counter
from contextlib import asynccontextmanager
import uuid

import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel

from services import llm, stt, conversation
from services import tts as tts_service

from core.config import (
    OLLAMA_URL_BASE,
    KOKORO_MODEL_PATH,
    KOKORO_VOICES_PATH,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)

from core.logging_config import logger, log_event


# ------------------------------
# APPLICATION LIFESPAN
# ------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Initialize application state
    app.state.tts = None
    app.state.whisper = None
    app.state.conversation_messages = []
    app.state.conversation_lock = asyncio.Lock()

    try:
        # Load Kokoro (TTS)
        startup_started = perf_counter()

        logger.info("Loading Bianca's voice...")

        app.state.tts = Kokoro(
            str(KOKORO_MODEL_PATH),
            str(KOKORO_VOICES_PATH)
        )

        log_event(
            "tts_loaded",
            startup_duration_ms=round(
                (perf_counter() - startup_started) * 1000, 2
            )
        )

        # Load Whisper (STT)
        startup_started = perf_counter()

        logger.info("Loading Whisper...")

        app.state.whisper = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )

        log_event(
            "stt_loaded",
            startup_duration_ms=round(
                (perf_counter() - startup_started) * 1000, 2
            )
        )

        yield

    finally:
        # Unload Kokoro
        if app.state.tts is not None:
            app.state.tts.voices.close()
            app.state.tts = None
            logger.info("Bianca's voice has been unloaded.")

        app.state.whisper = None


# ------------------------------
# APPLICATION SETUP
# ------------------------------

app = FastAPI(
    title="Bianca Server",
    lifespan=lifespan
)


# ------------------------------
# MIDDLEWARE
# ------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ------------------------------
# HEALTH
# ------------------------------

@app.get("/health")
def health():
    return {
        "status": "online",
        "assistant": "Bianca"
    }


@app.get("/ollama_health")
async def ollama_health():

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(OLLAMA_URL_BASE)

            return (
                response.status_code == 200
                and "Ollama" in response.text
            )

    except httpx.HTTPError:
        return False


# ------------------------------
# CHAT
# ------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.post("/chat")
async def chat(request: ChatRequest):

    messages = [
        {
            "role": message.role,
            "content": message.content
        }
        for message in request.messages
    ]

    return await llm.generate(messages)


# ------------------------------
# SPEAK
# ------------------------------

class SpeakRequest(BaseModel):
    text: str


@app.post("/speak")
def speak(request: SpeakRequest, http_request: Request):

    request_id = str(uuid.uuid4())[:8]

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    if http_request.app.state.tts is None:
        raise HTTPException(
            status_code=503,
            detail="Bianca's voice is not ready."
        )

    log_event(
        "speech_started",
        request_id=request_id,
        input_characters=len(request.text)
    )

    try:
        audio_bytes, _ = tts_service.generate(
            http_request.app.state.tts,
            request.text,
            request_id
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="bianca.wav"',
                "X-Request-ID": request_id,
            }
        )

    except Exception as error:

        log_event(
            "speech_failed",
            request_id=request_id,
            error=str(error)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {error}"
        )


# ------------------------------
# LISTEN
# ------------------------------

@app.post("/listen")
def listen(request: Request):

    if request.app.state.whisper is None:
        raise HTTPException(
            status_code=503,
            detail="Whisper is not ready."
        )

    return stt.transcribe(request.app.state.whisper)


# ------------------------------
# VOICE CONVERSATION
# ------------------------------

@app.post("/conversation")
async def voice_conversation(request: Request):

    whisper_model = request.app.state.whisper
    tts_model = request.app.state.tts

    if whisper_model is None:
        raise HTTPException(
            status_code=503,
            detail="Whisper is not ready."
        )

    if tts_model is None:
        raise HTTPException(
            status_code=503,
            detail="Bianca's voice is not ready."
        )

    # Prevent multiple conversations from running
    # simultaneously.

    async with request.app.state.conversation_lock:

        result = await conversation.process_turn(
            whisper_model,
            tts_model,
            request.app.state.conversation_messages
        )

        return result


# ------------------------------
# RESET CONVERSATION
# ------------------------------

@app.post("/conversation/reset")
async def reset_conversation(request: Request):

    async with request.app.state.conversation_lock:

        request.app.state.conversation_messages.clear()

    return {
        "status": "reset",
        "message": "Conversation history cleared."
    }