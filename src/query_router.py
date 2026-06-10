from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRoute:
    intent: str
    retrieval_mode: str
    use_rerank: bool
    top_k: int
    fetch_k: int
    reason: str


FACT_PATTERNS = [
    r"\d+",
    r"编号",
    r"号码",
    r"金额",
    r"日期",
    r"时间",
    r"多少",
    r"第几",
    r"哪一",
    r"准确率",
    r"召回率",
    r"分数",
    r"score",
    r"accuracy",
    r"date",
    r"number",
]

SUMMARY_PATTERNS = [
    r"总结",
    r"概括",
    r"主要",
    r"讲了什么",
    r"核心",
    r"overview",
    r"summary",
    r"main idea",
]

VISUAL_PATTERNS = [
    r"图",
    r"图片",
    r"截图",
    r"图表",
    r"柱状图",
    r"折线图",
    r"表格",
    r"figure",
    r"chart",
    r"image",
    r"screenshot",
    r"table",
]


def route_query(question: str) -> QueryRoute:
    normalized = question.strip().lower()
    if _matches(normalized, VISUAL_PATTERNS):
        return QueryRoute(
            intent="visual_or_table_lookup",
            retrieval_mode="hybrid",
            use_rerank=True,
            top_k=5,
            fetch_k=12,
            reason="问题涉及图片、图表或表格，优先使用混合检索召回 OCR/视觉描述片段。",
        )

    if _matches(normalized, FACT_PATTERNS):
        return QueryRoute(
            intent="fact_lookup",
            retrieval_mode="hybrid",
            use_rerank=True,
            top_k=5,
            fetch_k=10,
            reason="问题包含数字、字段或具体事实，混合检索更适合匹配专有名词和精确文本。",
        )

    if _matches(normalized, SUMMARY_PATTERNS):
        return QueryRoute(
            intent="summary",
            retrieval_mode="vector",
            use_rerank=True,
            top_k=6,
            fetch_k=12,
            reason="问题偏总结概括，向量检索更适合召回语义相关段落。",
        )

    return QueryRoute(
        intent="general_qa",
        retrieval_mode="hybrid",
        use_rerank=True,
        top_k=5,
        fetch_k=10,
        reason="默认使用混合检索，在语义召回和关键词匹配之间取得平衡。",
    )


def _matches(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
