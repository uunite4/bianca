from time import perf_counter
import uuid

import httpx

from ..config import OLLAMA_URL_BASE, OLLAMA_URL_CHAT, LLM_MODEL
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