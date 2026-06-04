from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import fitz

from src.ocr import ocr_image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PageText:
    source: str
    page: int
    text: str


def load_document(path: str | Path, enable_ocr: bool = False) -> list[PageText]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path, enable_ocr=enable_ocr)
    if suffix in {".txt", ".md"}:
        return _load_text(file_path)
    if suffix in IMAGE_SUFFIXES:
        return _load_image(file_path, enable_ocr=enable_ocr)

    allowed = "PDF, TXT, MD, PNG, JPG, JPEG, BMP, WEBP, TIF, TIFF"
    raise ValueError(f"Unsupported file type: {suffix}. Please upload {allowed}.")


def _load_pdf(path: Path, enable_ocr: bool = False) -> list[PageText]:
    pages: list[PageText] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text and enable_ocr:
                text = _ocr_pdf_page(page)
            if text:
                pages.append(PageText(source=path.name, page=index, text=text))
    return pages


def _load_text(path: Path) -> list[PageText]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [PageText(source=path.name, page=1, text=text)]


def _load_image(path: Path, enable_ocr: bool = False) -> list[PageText]:
    if not enable_ocr:
        raise ValueError("图片文件需要启用 OCR 后才能解析。请勾选“启用 OCR”。")

    text = ocr_image(path)
    if not text:
        return []
    return [PageText(source=path.name, page=1, text=text)]


def _ocr_pdf_page(page: fitz.Page) -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        pix.save(temp_path)
        return ocr_image(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
