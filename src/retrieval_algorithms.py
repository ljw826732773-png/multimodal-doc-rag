from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    words = re.findall(r"[a-z0-9]+", normalized)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    chinese_bigrams = [
        chinese[index : index + 2]
        for index in range(max(0, len(chinese) - 1))
    ]
    chinese_unigrams = list(chinese)
    return words + chinese_bigrams + chinese_unigrams


def bm25_search(
    query: str,
    records: list[dict[str, Any]],
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[dict[str, Any], float]]:
    if not records:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    doc_tokens = [tokenize(record.get("text", "")) for record in records]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    doc_freq = _document_frequency(doc_tokens)
    query_counter = Counter(query_terms)

    scored = []
    total_docs = len(records)
    for record, tokens, doc_length in zip(records, doc_tokens, doc_lengths):
        token_counts = Counter(tokens)
        score = 0.0
        for term, query_count in query_counter.items():
            frequency = token_counts.get(term, 0)
            if frequency == 0:
                continue

            idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length else 1.0
            term_score = idf * ((frequency * (k1 + 1)) / (frequency + k1 * length_norm))
            score += query_count * term_score

        if score > 0:
            scored.append((record, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - low) / (high - low) for value in values]


def select_mmr(
    hits: list[dict[str, Any]],
    top_k: int,
    lambda_mult: float = 0.72,
) -> list[dict[str, Any]]:
    if len(hits) <= top_k:
        return hits[:top_k]

    candidates = [dict(hit) for hit in hits]
    selected: list[dict[str, Any]] = []

    while candidates and len(selected) < top_k:
        best_index = 0
        best_score = float("-inf")

        for index, candidate in enumerate(candidates):
            relevance = float(candidate.get("score", 0.0))
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(_hit_similarity(candidate, item) for item in selected)

            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index

        chosen = candidates.pop(best_index)
        chosen["mmr_score"] = round(best_score, 6)
        selected.append(chosen)

    return selected


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for list_index, hits in enumerate(ranked_lists):
        for rank, hit in enumerate(hits, start=1):
            hit_id = str(hit.get("id"))
            if not hit_id:
                continue

            contribution = 1.0 / (rank_constant + rank)
            item = fused.setdefault(
                hit_id,
                {
                    **hit,
                    "rrf_score": 0.0,
                    "fusion_sources": [],
                    "score": 0.0,
                },
            )
            item["rrf_score"] += contribution
            item["fusion_sources"].append(
                {
                    "list": list_index,
                    "rank": rank,
                    "score": round(float(hit.get("score", 0.0)), 6),
                    "query_variant": hit.get("query_variant"),
                    "retrieval_mode": hit.get("retrieval_mode"),
                }
            )

            if float(hit.get("score", 0.0)) > float(item.get("_best_original_score", -1.0)):
                item.update(hit)
                item["_best_original_score"] = float(hit.get("score", 0.0))

    results = []
    for item in fused.values():
        item.pop("_best_original_score", None)
        results.append(item)

    results = sorted(results, key=lambda hit: hit["rrf_score"], reverse=True)
    normalized_scores = normalize_scores([float(hit["rrf_score"]) for hit in results])
    for hit, normalized_score in zip(results, normalized_scores):
        hit["score"] = normalized_score
        hit["rrf_score"] = round(float(hit["rrf_score"]), 6)

    return results[:top_k]


def _document_frequency(doc_tokens: list[list[str]]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for tokens in doc_tokens:
        frequency.update(set(tokens))
    return frequency


def _hit_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_embedding = left.get("_embedding")
    right_embedding = right.get("_embedding")
    if left_embedding is not None and right_embedding is not None:
        return _cosine(np.asarray(left_embedding, dtype=np.float32), np.asarray(right_embedding, dtype=np.float32))

    left_tokens = Counter(tokenize(left.get("text", "")))
    right_tokens = Counter(tokenize(right.get("text", "")))
    if not left_tokens or not right_tokens:
        return 0.0
    common = set(left_tokens) & set(right_tokens)
    dot = sum(left_tokens[token] * right_tokens[token] for token in common)
    left_norm = sum(value * value for value in left_tokens.values()) ** 0.5
    right_norm = sum(value * value for value in right_tokens.values()) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
