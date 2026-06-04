from __future__ import annotations

from functools import lru_cache
from pathlib import Path


class OCRUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise OCRUnavailableError(
            "OCR 依赖未安装。请先运行：pip install rapidocr-onnxruntime"
        ) from exc
    return RapidOCR()


def ocr_image(path: str | Path) -> str:
    engine = get_ocr_engine()
    result, _elapsed = engine(str(path))
    if not result:
        return ""
    return "\n".join(item[1] for item in result if len(item) >= 2 and item[1]).strip()
