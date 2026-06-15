from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieval_algorithms import tokenize


@dataclass(frozen=True)
class QueryVariant:
    query: str
    strategy: str
    reason: str


STOPWORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "does",
    "do",
    "did",
    "is",
    "are",
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "with",
    "about",
    "this",
    "that",
    "these",
    "those",
    "什么",
    "为什么",
    "怎么",
    "如何",
    "哪个",
    "哪些",
    "这个",
    "这些",
    "那个",
    "上面",
    "刚才",
}


def build_query_variants(
    question: str,
    contextual_query: str | None = None,
    conversation_context: str = "",
    max_variants: int = 4,
) -> list[QueryVariant]:
    candidates = [
        QueryVariant(question.strip(), "original", "Use the user's original question."),
    ]

    if contextual_query and contextual_query.strip() and contextual_query.strip() != question.strip():
        candidates.append(
            QueryVariant(
                contextual_query.strip(),
                "contextual",
                "Use recent conversation to resolve follow-up references.",
            )
        )

    keyword_query = build_keyword_query(question)
    if keyword_query:
        candidates.append(
            QueryVariant(
                keyword_query,
                "keywords",
                "Keep high-signal terms for BM25 and exact-match retrieval.",
            )
        )

    focused = build_focused_query(question, conversation_context)
    if focused:
        candidates.append(
            QueryVariant(
                focused,
                "focused",
                "Blend the current question with compact conversation clues.",
            )
        )

    return _dedupe(candidates)[:max_variants]


def build_keyword_query(question: str, max_terms: int = 12) -> str:
    tokens = tokenize(question)
    high_signal = []
    for token in tokens:
        if token in STOPWORDS or len(token.strip()) <= 1:
            continue
        if re.fullmatch(r"[a-z0-9]+", token) or re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            high_signal.append(token)
    return " ".join(_unique(high_signal)[:max_terms])


def build_focused_query(question: str, conversation_context: str, max_context_terms: int = 10) -> str:
    if not conversation_context.strip():
        return ""

    context_terms = []
    for token in tokenize(conversation_context):
        if token in STOPWORDS or len(token) <= 1:
            continue
        context_terms.append(token)

    compact_context = " ".join(_unique(context_terms)[:max_context_terms])
    if not compact_context:
        return ""
    return f"{question.strip()} {compact_context}".strip()


def _dedupe(variants: list[QueryVariant]) -> list[QueryVariant]:
    seen = set()
    result = []
    for variant in variants:
        normalized = " ".join(variant.query.lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(variant)
    return result


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
