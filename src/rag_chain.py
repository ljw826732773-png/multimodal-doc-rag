from __future__ import annotations

from src.llm_client import LLMError, generate_with_deepseek, generate_with_ollama


def build_retrieval_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "没有在当前知识库中检索到相关内容。"

    lines = [
        "我先给出基于检索片段的回答草稿。当前没有启用生成式大模型，所以这里会把最相关的证据整理出来。",
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
            "学习提示：这一步对应 RAG 里的 Retrieve，也就是先找资料。启用 DeepSeek 或 Ollama 后，会把这些片段作为上下文交给模型生成自然语言答案。",
        ]
    )
    return "\n".join(lines)


def build_llm_answer(
    question: str,
    contexts: list[dict],
    provider: str = "disabled",
    model: str = "deepseek-v4-flash",
    base_url: str = "https://api.deepseek.com",
    api_key: str = "",
) -> str:
    if provider == "disabled":
        return build_retrieval_answer(question, contexts)

    if not contexts:
        return "没有在当前知识库中检索到相关内容，无法基于文档回答。"

    prompt = build_rag_prompt(question, contexts)

    try:
        if provider == "deepseek":
            return generate_with_deepseek(
                prompt=prompt,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
        if provider == "ollama":
            return generate_with_ollama(prompt=prompt, model=model, base_url=base_url)
    except LLMError as exc:
        fallback = build_retrieval_answer(question, contexts)
        return f"{exc}\n\n已降级为检索结果草稿：\n\n{fallback}"

    raise ValueError(f"Unsupported provider: {provider}")


def build_rag_prompt(question: str, contexts: list[dict]) -> str:
    evidence = []
    for index, item in enumerate(contexts, start=1):
        source = item.get("source", "unknown")
        page = item.get("page", "?")
        chunk_index = item.get("chunk_index", "?")
        text = item.get("text", "")
        evidence.append(
            f"[{index}] 来源：{source}，第 {page} 页，chunk {chunk_index}\n{text}"
        )

    joined_evidence = "\n\n".join(evidence)
    return f"""你是一个严谨的文档问答助手。

请只根据给定资料回答问题，不要编造资料中没有的信息。
如果资料中没有答案，请直接说“根据当前文档无法确定”。
回答要清晰、简洁，并在关键结论后标注引用编号，例如：[1]。

资料：
{joined_evidence}

问题：
{question}

答案："""
