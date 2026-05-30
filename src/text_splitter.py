from __future__ import annotations

from dataclasses import dataclass

from src.document_loader import PageText


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    chunk_index: int
    text: str


def split_pages(
    pages: list[PageText],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[Chunk] = []
    for page in pages:
        normalized = _normalize_text(page.text)
        start = 0
        chunk_index = 0

        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            text = normalized[start:end].strip()
            if text:
                chunk_id = f"{page.source}:p{page.page}:c{chunk_index}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        source=page.source,
                        page=page.page,
                        chunk_index=chunk_index,
                        text=text,
                    )
                )
                chunk_index += 1

            if end == len(normalized):
                break
            start = max(0, end - overlap)

    return chunks


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
