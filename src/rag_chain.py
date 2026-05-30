from __future__ import annotations


def build_retrieval_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "没有在当前知识库中检索到相关内容。"

    lines = [
        "我先给出基于检索片段的回答草稿。当前版本还没有接入生成式大模型，所以这里会把最相关的证据整理出来，后续会替换成 LLM 生成答案。",
        "",
        f"问题：{question}",
        "",
        "最相关依据：",
    ]

    for index, item in enumerate(contexts[:3], start=1):
        source = item.get("source", "unknown")
        page = item.get("page", "?")
        score = item.get("score", 0.0)
        preview = item.get("text", "").replace("\n", " ")
        if len(preview) > 260:
            preview = preview[:260].rstrip() + "..."
        lines.append(f"{index}. {source} 第 {page} 页，相似度 {score:.3f}：{preview}")

    lines.extend(
        [
            "",
            "学习提示：这一步对应 RAG 里的 Retrieve，也就是先找资料。下一步接入 LLM 后，会把这些片段作为上下文交给模型生成自然语言答案。",
        ]
    )
    return "\n".join(lines)
