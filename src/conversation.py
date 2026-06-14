from __future__ import annotations

from typing import Any


FOLLOW_UP_HINTS = [
    "它",
    "这个",
    "这些",
    "上述",
    "上面",
    "刚才",
    "继续",
    "再",
    "that",
    "it",
    "those",
    "above",
    "previous",
]


def build_conversation_context(history: list[dict[str, Any]], max_turns: int = 3) -> str:
    turns = history[-max_turns:]
    lines: list[str] = []

    for index, item in enumerate(turns, start=1):
        question = _compact(item.get("question", ""), limit=220)
        answer = _compact(item.get("answer", ""), limit=360)
        if not question and not answer:
            continue
        lines.append(f"Turn {index} question: {question}")
        lines.append(f"Turn {index} answer: {answer}")

    return "\n".join(lines)


def build_contextual_query(question: str, history: list[dict[str, Any]], max_turns: int = 3) -> str:
    context = build_conversation_context(history, max_turns=max_turns)
    if not context:
        return question

    if _looks_like_follow_up(question):
        return f"{context}\nCurrent follow-up question: {question}"

    return f"{question}\nRecent conversation for disambiguation:\n{context}"


def _looks_like_follow_up(question: str) -> bool:
    normalized = question.strip().lower()
    return any(hint in normalized for hint in FOLLOW_UP_HINTS) or len(normalized) <= 12


def _compact(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."
