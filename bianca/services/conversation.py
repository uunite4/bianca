from time import perf_counter
from io import BytesIO
import asyncio
import sounddevice as sd
import soundfile as sf
import re
import uuid
from collections import deque

from starlette.concurrency import run_in_threadpool

from . import llm, stt
from . import tts as tts_service
from .chunking import sentence_stream

from ..logging_config import log_event
from ..config import (
    GATE_LOOKAHEAD_CHUNKS,
    STREAM_OPEN_SECONDS,
    START_MARGIN_SECONDS,
    MAX_START_WAIT_SECONDS,
    TTS_IDLE_RATIO,
    TTS_SAMPLE_RATE,
)

END_PHRASES = {
    "thanks",
    "thank you",
    "thanks a lot",
    "thanks so much",
    "thank you very much",
    "thank you so much",
}


def is_end_conversation_phrase(text):
    """Check whether the user wants to end the conversation."""

    normalized = re.sub(r"[^\w\s]", "", text.lower())
    words = normalized.split()

    # Ignore Bianca's name at the start or the end ("Bianca, thank you.").
    if words and words[0] == "bianca":
        words = words[1:]

    if words and words[-1] == "bianca":
        words = words[:-1]

    return " ".join(words) in END_PHRASES

def clean_transcription(text: str) -> str:
    """
    Remove Bianca's wake word if it appears at the
    beginning of the transcription.
    """

    return re.sub(
        r"^\s*bianca\b[\s,.:;!?-]*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()

async def process_turn(
    whisper_model,
    tts_model,
    messages: list[dict],
    audio,
):
    request_started = perf_counter()

    # ------------------------------
    # 1. LISTEN AND TRANSCRIBE
    # ------------------------------

    stt_started = perf_counter()

    transcription = await run_in_threadpool(
        stt.transcribe_audio,
        whisper_model,
        audio,
    )

    raw_text = transcription["text"]
    language = transcription["language"]

    # Check for an end-of-conversation phrase.
    end_conversation = is_end_conversation_phrase(raw_text)

    user_text = clean_transcription(raw_text)

    if not user_text:
        log_event(
            "conversation_empty",
            stt_duration_ms=round(
                (perf_counter() - stt_started) * 1000,
                2
            )
        )

        return {
            "status": "empty",
            "text": "",
            "reply": None,
        }

    log_event(
        "conversation_transcribed",
        text=user_text,
        language=language,
        stt_duration_ms=round(
            (perf_counter() - stt_started) * 1000,
            2
        ),
        audio_seconds=round(len(audio) / 16000, 2),
        end_conversation=end_conversation,
        history_messages=len(messages),
    )

    # ------------------------------
    # 2. STREAM REPLY -> SPEECH -> PLAYBACK
    # ------------------------------

    request_id = str(uuid.uuid4())[:8]

    conversation_messages = [
        *messages,
        {
            "role": "user",
            "content": user_text,
        }
    ]

    raw_parts = []

    async def record(pieces):
        # Pass every piece through unchanged, but keep a copy of the raw text.
        async for piece in pieces:
            raw_parts.append(piece)
            yield piece

    pieces = record(llm.generate_stream(conversation_messages, request_id))
    sentences = sentence_stream(pieces)

    reply = await speak_stream(
        tts_model,
        sentences,
        request_started,
    )

    # ------------------------------
    # 5. UPDATE HISTORY
    # ------------------------------

    messages.append({
        "role": "user",
        "content": user_text,
    })

    messages.append({
        "role": "assistant",
        "content": "".join(raw_parts),
    })

    total_duration_ms = round(
        (perf_counter() - request_started) * 1000,
        2
    )

    log_event(
        "conversation_turn_completed",
        total_duration_ms=total_duration_ms
    )

    return {
        "status": "end-conversation" if end_conversation else "completed",
        "text": user_text,
        "language": language,
        "reply": reply,
        "request_id": request_id,
        "total_duration_ms": total_duration_ms,
    }


def play_audio(audio, sample_rate):

    sd.play(
        audio,
        samplerate=sample_rate
    )

    sd.wait()

def _open_output_stream():
    """Open and start the speaker (takes about half a second)."""

    stream = sd.OutputStream(
        samplerate=TTS_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    stream.start()

    return stream

async def speak_stream(tts_model, sentences, request_started):
    """
    Speak a stream of sentences. Synthesis and playback run as separate
    tasks, and the first word is held back until the buffered audio can
    cover the synthesis of the upcoming chunk(s).
    Returns the full text that was spoken.
    """

    text_queue = asyncio.Queue()
    audio_queue = asyncio.Queue()

    # Length (in characters) of every chunk waiting for, or currently in,
    # synthesis. Used to predict what's coming next.
    backlog = deque()

    # Measured speeds, updated after every synthesized chunk:
    #   ratio        = seconds of synthesis per second of audio
    #   sec_per_char = seconds of audio per character of text
    speed = {"ratio": 1.0, "sec_per_char": 0.07}
    llm_done = {"value": False}

    def at_ms():
        return round((perf_counter() - request_started) * 1000, 2)

    async def tts_worker():
        index = 0

        try:
            while True:
                wait_started = perf_counter()
                text = await text_queue.get()
                idle_wait_ms = round(
                    (perf_counter() - wait_started) * 1000, 2
                )

                if text is None:
                    return

                synth_started = perf_counter()

                samples, sample_rate = await run_in_threadpool(
                    tts_service.synthesize,
                    tts_model,
                    text,
                )

                synth_seconds = perf_counter() - synth_started
                audio_seconds = len(samples) / sample_rate

                speed["ratio"] = synth_seconds / audio_seconds
                speed["sec_per_char"] = audio_seconds / len(text)

                if backlog:
                    backlog.popleft()

                log_event(
                    "pipeline_tts_done",
                    chunk=index,
                    text=text[:60],
                    tts_idle_wait_ms=idle_wait_ms,
                    synth_ms=round(synth_seconds * 1000, 2),
                    audio_seconds=round(audio_seconds, 2),
                    ratio=round(speed["ratio"], 2),
                    at_ms=at_ms(),
                )

                await audio_queue.put((index, samples, sample_rate))
                index += 1
        finally:
            await audio_queue.put(None)

    async def player():
        stream = None
        play_end = None
        chunks_played = 0

        held = []          # chunks waiting for the start gate
        gate_open = False
        first_ready = None

        async def play_chunk(item, player_wait_ms):
            nonlocal stream, play_end, chunks_played

            index, samples, sample_rate = item
            duration = len(samples) / sample_rate

            if stream is None:
                open_started = perf_counter()

                stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                )
                stream.start()

                if sample_rate != TTS_SAMPLE_RATE:
                    log_event(
                    "pipeline_sample_rate_mismatch",
                    expected=TTS_SAMPLE_RATE,
                    got=sample_rate,
                    )

            # Estimate how long the room was silent before this chunk.
            now = perf_counter()
            gap_before_ms = 0.0

            if play_end is None or now > play_end:
                if play_end is not None:
                    gap_before_ms = round((now - play_end) * 1000, 2)

                play_end = now

            play_end += duration

            chunk_start_ms = at_ms()

            if index == 0:
                log_event(
                    "time_to_first_audio",
                    time_to_first_audio_ms=chunk_start_ms,
                )

            underflow = await run_in_threadpool(
                stream.write,
                samples.reshape(-1, 1),
            )

            chunks_played += 1

            log_event(
                "pipeline_chunk_play",
                chunk=index,
                audio_seconds=round(duration, 2),
                player_wait_ms=player_wait_ms,
                gap_before_ms=gap_before_ms,
                underflow=bool(underflow),
                chunk_start_ms=chunk_start_ms,
            )

        try:
            open_started = perf_counter()
            stream = await run_in_threadpool(_open_output_stream)

            log_event(
                "pipeline_stream_opened",
                open_ms=round((perf_counter() - open_started) * 1000, 2),
                at_ms=at_ms(),
            )

            while True:
                wait_started = perf_counter()
                item = await audio_queue.get()
                player_wait_ms = round(
                    (perf_counter() - wait_started) * 1000, 2
                )

                if item is None:
                    break

                if gate_open:
                    await play_chunk(item, player_wait_ms)
                    continue

                # The first word is still being held back.
                held.append((item, player_wait_ms))

                if first_ready is None:
                    first_ready = perf_counter()

                buffered = sum(
                    len(samples) / rate
                    for (_, samples, rate), _wait in held
                )

                upcoming_chars = max(
                    list(backlog)[:GATE_LOOKAHEAD_CHUNKS],
                    default=0,
                )

                ratio = (
                    TTS_IDLE_RATIO
                    if llm_done["value"]
                    else speed["ratio"]
                )

                predicted = ratio * speed["sec_per_char"] * upcoming_chars

                needed = (
                    predicted
                    - STREAM_OPEN_SECONDS
                    + START_MARGIN_SECONDS
                )

                waited = perf_counter() - first_ready

                if upcoming_chars == 0:
                    reason = "nothing_queued"
                elif buffered >= needed:
                    reason = "buffer_ok"
                elif waited >= MAX_START_WAIT_SECONDS:
                    reason = "max_wait"
                else:
                    continue

                gate_open = True

                log_event(
                    "pipeline_start_gate",
                    reason=reason,
                    buffered_s=round(buffered, 2),
                    needed_s=round(needed, 2),
                    waited_ms=round(waited * 1000, 2),
                    held_chunks=len(held),
                    at_ms=at_ms(),
                )

                for held_item, held_wait in held:
                    await play_chunk(held_item, held_wait)

                held.clear()

            # The reply ended before the gate opened: play what we have.
            for held_item, held_wait in held:
                await play_chunk(held_item, held_wait)

        finally:
            if stream is not None:
                drain_started = perf_counter()

                # stop() blocks until everything has finished playing.
                await run_in_threadpool(stream.stop)
                stream.close()

                log_event(
                    "pipeline_playback_drained",
                    drain_ms=round(
                        (perf_counter() - drain_started) * 1000, 2
                    ),
                    chunks=chunks_played,
                    at_ms=at_ms(),
                )

    tts_task = asyncio.create_task(tts_worker())
    player_task = asyncio.create_task(player())

    spoken = []
    sentence_index = 0

    try:
        async for sentence in sentences:
            log_event(
                "pipeline_sentence_ready",
                chunk=sentence_index,
                chars=len(sentence),
                text=sentence[:60],
                at_ms=at_ms(),
            )

            sentence_index += 1
            spoken.append(sentence)
            backlog.append(len(sentence))
            await text_queue.put(sentence)
        llm_done["value"] = True
    finally:
        await text_queue.put(None)
        await asyncio.gather(tts_task, player_task)

    return " ".join(spoken)