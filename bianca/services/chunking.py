import re

# Chunks shorter than this get merged into the next sentence, so Bianca
# never speaks tiny fragments like "Sure!" or "Dr." on their own.
MIN_CHUNK_CHARS = 20

# Long chunks take too long to synthesize while the LLM is still running
# (TTS is close to real time then), which leaves silence in the middle of
# the reply. Anything longer than this is split at a comma, colon, etc.
MAX_CHUNK_CHARS = 80

# The first chunk decides how soon she starts talking, so a long first
# sentence is split at its commas into pieces of about this size.
FIRST_CHUNK_MAX_CHARS = 45

_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:])\s+")

# A boundary is whitespace after . ! ? or a line break.
# "3.5" isn't split because there's no whitespace after the dot.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

_MARKDOWN = re.compile(r"[*_`#>]+")


def clean_for_speech(text: str) -> str:
    """Remove markdown symbols the TTS would stumble over."""

    text = _MARKDOWN.sub("", text)
    text = re.sub(r"^\s*[-•]\s+", "", text)

    return " ".join(text.split())


def split_long_chunk(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split text longer than `limit` at clause boundaries (after , ; :).
    A clause is never cut in the middle.
    """

    if len(text) <= limit:
        return [text]

    pieces = []
    current = ""

    for clause in _CLAUSE_BOUNDARY.split(text):
        candidate = f"{current} {clause}".strip()

        if current and len(candidate) > limit:
            pieces.append(current)
            current = clause
        else:
            current = candidate

    if current:
        pieces.append(current)

    # A tiny last piece sounds abrupt, so merge it into the previous one.
    if len(pieces) > 1 and len(pieces[-1]) < MIN_CHUNK_CHARS:
        last = pieces.pop()
        pieces[-1] = f"{pieces[-1]} {last}"

    return pieces

async def sentence_stream(pieces):
    """
    Take a stream of text pieces and yield speakable chunks
    (usually whole sentences) as soon as each one is complete.
    """

    buffer = ""
    pending = ""
    first_chunk = True

    async for piece in pieces:
        buffer += piece

        while True:
            match = _BOUNDARY.search(buffer)

            if match is None:
                break

            sentence = clean_for_speech(buffer[:match.start()])
            buffer = buffer[match.end():]

            if not sentence:
                continue

            pending = f"{pending} {sentence}".strip()

            if len(pending) >= MIN_CHUNK_CHARS:
                limit = (
                    FIRST_CHUNK_MAX_CHARS if first_chunk else MAX_CHUNK_CHARS
                )
                first_chunk = False

                for piece in split_long_chunk(pending, limit):
                    yield piece

                pending = ""

    # The stream ended: speak whatever is left.
    tail = clean_for_speech(f"{pending} {buffer}")

    if tail:
        limit = FIRST_CHUNK_MAX_CHARS if first_chunk else MAX_CHUNK_CHARS

        for piece in split_long_chunk(tail, limit):
            yield piece