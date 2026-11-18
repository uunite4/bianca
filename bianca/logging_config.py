import logging
import sys

class BiancaFormatter(logging.Formatter):

    COLORS = {
        "INFO": "\033[92m",      # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "DEBUG": "\033[90m",     # Grey
    }

    RESET = "\033[0m"

    EVENT_NAMES = {
        "tts_loaded": "VOICE LOADED",
        "chat_started": "CHAT STARTED",
        "chat_completed": "CHAT COMPLETED",
        "chat_failed": "CHAT FAILED",
        "speech_started": "SPEECH STARTED",
        "speech_completed": "SPEECH GENERATED",
        "speech_failed": "SPEECH FAILED",
        "stt_completed": "TRANSCRIPTION",
    }

    # Fields that are printed in a special way (everything else is
    # printed automatically as key=value).
    HANDLED_FIELDS = {
        "total_duration_ms",
        "startup_duration_ms",
        "generation_duration_ms",
        "tokens_per_second",
        "text",
        "error",
    }

    def format(self, record):
        timestamp = (
            f"{self.formatTime(record, '%H:%M:%S')}"
            f".{int(record.msecs):03d}"
        )

        data = getattr(record, "event_data", {})
        event = record.getMessage()

        name = self.EVENT_NAMES.get(event, event.upper())
        color = self.COLORS.get(record.levelname, "")

        details = []

        duration = (
            data.get("total_duration_ms")
            or data.get("startup_duration_ms")
            or data.get("generation_duration_ms")
        )

        if duration is not None:
            details.append(f"{duration / 1000:.2f}s")

        if "tokens_per_second" in data:
            details.append(f"{data['tokens_per_second']} tokens/s")

        # Print every other field as key=value so nothing is hidden.
        for key, value in data.items():
            if key not in self.HANDLED_FIELDS:
                details.append(f"{key}={value}")

        if "text" in data:
            details.append(f'"{data["text"]}"')

        if "error" in data:
            details.append(f"Error: {data['error']}")

        details_text = "  ".join(details)

        message = f"{timestamp}  {name:<28} {details_text}"

        if sys.stderr.isatty():
            return f"{color}{message}{self.RESET}"

        return message


def configure_logging():
    logger = logging.getLogger("bianca")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(BiancaFormatter())

    logger.addHandler(handler)

    return logger


logger = configure_logging()


def log_event(event: str, **data):
    logger.info(event, extra={"event_data": data})