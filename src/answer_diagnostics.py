from __future__ import annotations

import re
from typing import Any

from src.reranker import lexical_similarity


def build_answer_diagnostics(question: str, answer: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    if not hits:
        return {
            "confidence": "low",
            "confidence_score": 0.0,
            "top_score": 0.0,
            "average_score": 0.0,
            "answer_coverage": 0.0,
            "source_count": 0,
            "warnings": ["No retrieved evidence was found."],
            "evidence_rows": [],
        }

    scores = [float(hit.get("score", 0.0)) for hit in hits]
    source_count = len({hit.get("source") for hit in hits if hit.get("source")})
    top_score = max(scores)
    average_score = sum(scores) / len(scores)
    answer_coverage = _answer_coverage(answer, hits)
    diversity_score = min(1.0, source_count / 3)

    confidence_score = _clamp(
        0.4 * top_score + 0.35 * answer_coverage + 0.15 * average_score + 0.1 * diversity_score
    )
    confidence = _confidence_label(confidence_score)
    warnings = _build_warnings(
        top_score=top_score,
        average_score=average_score,
        answer_coverage=answer_coverage,
        source_count=source_count,
        hit_count=len(hits),
    )

    return {
        "confidence": confidence,
        "confidence_score": round(confidence_score, 4),
        "top_score": round(top_score, 4),
        "average_score": round(average_score, 4),
        "answer_coverage": round(answer_coverage, 4),
        "source_count": source_count,
        "warnings": warnings,
        "evidence_rows": _evidence_rows(question, answer, hits),
    }


def _answer_coverage(answer: str, hits: list[dict[str, Any]]) -> float:
    sentences = _sentences(answer)
    if not sentences:
        return 0.0

    evidence_texts = [hit.get("text", "") for hit in hits]
    covered = []
    for sentence in sentences:
        best = max(lexical_similarity(sentence, evidence) for evidence in evidence_texts)
        covered.append(best)
    return sum(covered) / len(covered)


def _evidence_rows(question: str, answer: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, hit in enumerate(hits, start=1):
        rows.append(
            {
                "rank": rank,
                "source": hit.get("source", "unknown"),
                "page": hit.get("page", "?"),
                "chunk": hit.get("chunk_index", "?"),
                "retrieval_score": round(float(hit.get("score", 0.0)), 4),
                "question_overlap": round(lexical_similarity(question, hit.get("text", "")), 4),
                "answer_overlap": round(lexical_similarity(answer, hit.get("text", "")), 4),
            }
        )
    return rows


def _build_warnings(
    top_score: float,
    average_score: float,
    answer_coverage: float,
    source_count: int,
    hit_count: int,
) -> list[str]:
    warnings = []
    if top_score < 0.25:
        warnings.append("Top evidence score is low; the answer may not be well supported.")
    if average_score < 0.18:
        warnings.append("Retrieved chunks are weak on average; try a clearer question or larger Fetch-K.")
    if answer_coverage < 0.12:
        warnings.append("Answer text has low lexical overlap with evidence; check for unsupported claims.")
    if hit_count >= 3 and source_count <= 1:
        warnings.append("Evidence comes from a single source; cross-document confirmation is limited.")
    return warnings


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|[\n\r]+", text)
    return [part.strip() for part in parts if len(part.strip()) >= 8]


def _confidence_label(score: float) -> str:
    if score >= 0.55:
        return "high"
    if score >= 0.32:
        return "medium"
    return "low"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
