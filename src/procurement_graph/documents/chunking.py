"""Document text chunking primitives for contract evidence.

This module does not fetch or OCR documents. It defines the stable chunk schema
that parsers should emit after a document has been downloaded or converted to
text by a cheap local toolchain.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Iterable


HEADING_RE = re.compile(
    r"^\s*((?:\d+(?:\.\d+)*|[A-Z])[\).]?\s+)?"
    r"(award criteria|evaluation criteria|specification|scope|contract terms|"
    r"payment|pricing|termination|insurance|data protection|tupe|social value|"
    r"delivery|implementation|key performance|kpi|service level|sla)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    url_hash: str
    ocid: str
    source: str
    document_type: str
    url: str
    text: str
    page_number: int | None = None
    heading: str = ""
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def stable_chunk_id(url_hash: str, page_number: int | None, char_start: int, char_end: int) -> str:
    raw = f"{url_hash}|{page_number or ''}|{char_start}|{char_end}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def normalise_document_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text or "")))


def iter_clause_blocks(text: str) -> Iterable[tuple[str, int, int, str]]:
    """Yield rough clause blocks with heading hints and character offsets."""
    clean = normalise_document_text(text)
    if not clean:
        return

    lines = clean.splitlines()
    cursor = 0
    current_heading = ""
    block_lines: list[str] = []
    block_start = 0

    def flush(end: int):
        nonlocal block_lines, block_start
        block_text = "\n".join(block_lines).strip()
        if block_text:
            yield block_text, block_start, end, current_heading
        block_lines = []

    for line in lines:
        line_start = cursor
        cursor += len(line) + 1
        if HEADING_RE.search(line):
            yield from flush(line_start)
            current_heading = line.strip()[:200]
            block_start = line_start
            block_lines = [line]
            continue
        if not block_lines:
            block_start = line_start
        block_lines.append(line)

    yield from flush(len(clean))


def chunk_text(
    *,
    text: str,
    url_hash: str,
    ocid: str,
    source: str,
    document_type: str,
    url: str,
    page_number: int | None = None,
    max_tokens: int = 650,
    overlap_tokens: int = 80,
) -> list[DocumentChunk]:
    """Chunk text into clause-aware windows.

    Large clause blocks are split into word windows. Offsets for split windows
    are approximate but stable enough for audit and citation.
    """
    chunks: list[DocumentChunk] = []
    for block_text, block_start, block_end, heading in iter_clause_blocks(text):
        words = block_text.split()
        if len(words) <= max_tokens:
            chunks.append(
                DocumentChunk(
                    chunk_id=stable_chunk_id(url_hash, page_number, block_start, block_end),
                    url_hash=url_hash,
                    ocid=ocid,
                    source=source,
                    document_type=document_type,
                    url=url,
                    text=block_text,
                    page_number=page_number,
                    heading=heading,
                    char_start=block_start,
                    char_end=block_end,
                    token_count=estimate_tokens(block_text),
                )
            )
            continue

        step = max(1, max_tokens - overlap_tokens)
        for start_word in range(0, len(words), step):
            window_words = words[start_word : start_word + max_tokens]
            if not window_words:
                continue
            window_text = " ".join(window_words)
            approx_start = block_start + len(" ".join(words[:start_word]))
            approx_end = min(block_end, approx_start + len(window_text))
            chunks.append(
                DocumentChunk(
                    chunk_id=stable_chunk_id(url_hash, page_number, approx_start, approx_end),
                    url_hash=url_hash,
                    ocid=ocid,
                    source=source,
                    document_type=document_type,
                    url=url,
                    text=window_text,
                    page_number=page_number,
                    heading=heading,
                    char_start=approx_start,
                    char_end=approx_end,
                    token_count=estimate_tokens(window_text),
                )
            )
            if start_word + max_tokens >= len(words):
                break
    return chunks


__all__ = [
    "DocumentChunk",
    "chunk_text",
    "estimate_tokens",
    "normalise_document_text",
    "stable_chunk_id",
]

