from __future__ import annotations

from functools import lru_cache
import hashlib
import math

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    try:
        model = get_embedding_model(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        array = np.asarray(embeddings, dtype=np.float32)
        return array.tolist()
    except Exception:
        return [_hash_embedding(text) for text in texts]


def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    tokens = _char_ngrams(text)
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8", errors="ignore")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _char_ngrams(text: str, n: int = 2) -> list[str]:
    compact = "".join(text.lower().split())
    if len(compact) <= n:
        return [compact] if compact else []
    return [compact[index : index + n] for index in range(len(compact) - n + 1)]
