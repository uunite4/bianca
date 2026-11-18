from time import perf_counter
import uuid

import httpx
import json

from ..config import (
    OLLAMA_URL_BASE,
    OLLAMA_URL_CHAT,
    LLM_MODEL,
    LLM_NUM_THREAD
)
from ..logging_config import log_event

class LLMError(Exception):
    """Raised when the LLM request fails or returns nothing."""

async def generate(messages: list[dict]):

    request_id = str(uuid.uuid4())[:8]
    request_started = perf_counter()

    if not messages:
        raise ValueError("At least one message is required")

    log_event(
        "chat_started",
        request_id=request_id,
        messages_count=len(messages),
        input_characters=sum(len(message["content"]) for message in messages),
    )

    ollama_started = perf_counter()

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                OLLAMA_URL_BASE + OLLAMA_URL_CHAT,
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": -1,
                },
            )
            response.raise_for_status()
            data = response.json()

    except httpx.HTTPError as error:
        log_event(
            "chat_failed",
            request_id=request_id,
            error=str(error),
            duration_ms=round((perf_counter() - request_started) * 1000, 2),
        )

        raise LLMError(f"Could not reach Ollama: {error}") from error

    ollama_http_duration_ms = round(
        (perf_counter() - ollama_started) * 1000, 2
    )

    reply = data.get("message", {}).get("content", "")
    date = data.get("created_at")

    if not reply:
        log_event(
            "chat_failed",
            request_id=request_id,
            error="Ollama returned an empty response",
            ollama_http_duration_ms=ollama_http_duration_ms
        )

        raise LLMError("Ollama returned an empty response.")

    ollama_stats = {
        "load_duration_ms": round(data.get("load_duration", 0) / 1_000_000, 2),
        "prompt_eval_duration_ms": round(data.get("prompt_eval_duration", 0) / 1_000_000, 2),
        "generation_duration_ms": round(data.get("eval_duration", 0) / 1_000_000, 2),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "generated_tokens": data.get("eval_count", 0),
    }

    generation_duration = ollama_stats["generation_duration_ms"]
    generated_tokens = ollama_stats["generated_tokens"]

    tokens_per_second = (
        round(generated_tokens / (generation_duration / 1000), 2)
        if generation_duration > 0
        else 0
    )

    total_duration_ms = round(
        (perf_counter() - request_started) * 1000, 2
    )

    log_event(
        "chat_completed",
        request_id=request_id,
        model=LLM_MODEL,
        ollama_http_duration_ms=ollama_http_duration_ms,
        total_duration_ms=total_duration_ms,
        reply_characters=len(reply),
        tokens_per_second=tokens_per_second,
        **ollama_stats,
    )

    duration = round(perf_counter() - request_started, 2)

    return {
        "reply": reply,
        "date": date,
        "duration": duration,
        "model": LLM_MODEL,
        "request_id": request_id,
        "metrics": {
            "ollama_http_duration_ms": ollama_http_duration_ms,
            "total_duration_ms": total_duration_ms,
            **ollama_stats,
            "tokens_per_second": tokens_per_second
        }
    }

async def generate_stream(messages: list[dict], request_id: str):
    """
    Stream the reply from Ollama, yielding text pieces as they are generated.
    """

    if not messages:
        raise ValueError("At least one message is required")

    started = perf_counter()
    first_piece_ms = None
    received_text = False

    log_event(
        "chat_stream_started",
        request_id=request_id,
        messages_count=len(messages),
        input_characters=sum(len(message["content"]) for message in messages),
    )

    options = {}
    if LLM_NUM_THREAD:
        options["num_thread"] = LLM_NUM_THREAD

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                OLLAMA_URL_BASE + OLLAMA_URL_CHAT,
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": -1,
                    "options": options,
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)

                    piece = data.get("message", {}).get("content", "")

                    if piece:
                        if first_piece_ms is None:
                            first_piece_ms = round(
                                (perf_counter() - started) * 1000, 2
                            )

                        received_text = True
                        yield piece

                    if data.get("done"):
                        generation_ms = data.get("eval_duration", 0) / 1_000_000
                        tokens = data.get("eval_count", 0)

                        log_event(
                            "chat_stream_completed",
                            request_id=request_id,
                            model=LLM_MODEL,
                            first_piece_ms=first_piece_ms,
                            total_duration_ms=round(
                                (perf_counter() - started) * 1000, 2
                            ),
                            load_duration_ms=round(
                                data.get("load_duration", 0) / 1_000_000, 2
                            ),
                            prompt_eval_duration_ms=round(
                                data.get("prompt_eval_duration", 0) / 1_000_000, 2
                            ),
                            generated_tokens=tokens,
                            tokens_per_second=(
                                round(tokens / (generation_ms / 1000), 2)
                                if generation_ms > 0
                                else 0
                            ),
                        )
                        break

    except httpx.HTTPError as error:
        log_event(
            "chat_failed",
            request_id=request_id,
            error=str(error),
        )

        raise LLMError(f"Could not reach Ollama: {error}") from error

    if not received_text:
        log_event(
            "chat_failed",
            request_id=request_id,
            error="Ollama returned an empty response",
        )

        raise LLMError("Ollama returned an empty response.")


async def warm_up():
    """Load the model now, so the first real question is not slow."""

    # Keep this identical to generate_stream, otherwise Ollama may
    # reload the model when the real request arrives.
    options = {}

    if LLM_NUM_THREAD:
        options["num_thread"] = LLM_NUM_THREAD

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            await client.post(
                OLLAMA_URL_BASE + OLLAMA_URL_CHAT,
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                    "keep_alive": -1,
                    "options": {**options, "num_predict": 1},
                },
            )
    except httpx.HTTPError as error:
        log_event("llm_warmup_failed", error=str(error))