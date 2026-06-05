from __future__ import annotations

import re
from collections import Counter


def rerank_hits(question: str, hits: list[dict], top_k: int) -> list[dict]:
    if not hits:
        return []

    question_terms = _terms(question)
    reranked = []
    for hit in hits:
        text_terms = _terms(hit.get("text", ""))
        lexical_score = _cosine(question_terms, text_terms)
        vector_score = float(hit.get("score", 0.0))
        combined_score = 0.75 * vector_score + 0.25 * lexical_score
        enriched = dict(hit)
        enriched["vector_score"] = vector_score
        enriched["rerank_score"] = combined_score
        enriched["score"] = combined_score
        reranked.append(enriched)

    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]


def lexical_similarity(left: str, right: str) -> float:
    return _cosine(_terms(left), _terms(right))


def _terms(text: str) -> Counter[str]:
    normalized = text.lower()
    words = re.findall(r"[a-z0-9]+", normalized)
    compact = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    char_bigrams = [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]
    return Counter(words + char_bigrams)


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
