from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class PageText:
    source: str
    page: int
    text: str


def load_document(path: str | Path) -> list[PageText]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix in {".txt", ".md"}:
        return _load_text(file_path)

    raise ValueError(f"Unsupported file type: {suffix}. Please upload PDF, TXT, or MD.")


def _load_pdf(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(source=path.name, page=index, text=text))
    return pages


def _load_text(path: Path) -> list[PageText]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [PageText(source=path.name, page=1, text=text)]
