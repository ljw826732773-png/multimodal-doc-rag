from __future__ import annotations

from src.llm_client import LLMError, generate_with_deepseek, generate_with_ollama


def build_retrieval_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "没有在当前知识库中检索到相关内容。"

    lines = [
        "当前没有启用生成式大模型，先展示检索到的依据。",
        "",
        f"问题：{question}",
        "",
        "结论：",
        "请启用 DeepSeek API 后生成完整答案；当前只能展示最相关的文档片段。",
        "",
        "依据：",
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
            "引用来源：",
            "见右侧命中的原文片段。启用 DeepSeek 或 Ollama 后，这些片段会作为上下文交给模型生成自然语言答案。",
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
    conversation_context: str = "",
) -> str:
    if provider == "disabled":
        return build_retrieval_answer(question, contexts)

    if not contexts:
        return "没有在当前知识库中检索到相关内容，无法基于文档回答。"

    prompt = build_rag_prompt(question, contexts, conversation_context=conversation_context)

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


def build_rag_prompt(question: str, contexts: list[dict], conversation_context: str = "") -> str:
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
    conversation_block = ""
    if conversation_context.strip():
        conversation_block = f"""
最近对话：
{conversation_context}

请只把最近对话用于理解当前问题的指代关系，不要把对话内容当作新的文档事实。
"""
    return f"""你是一个严谨的文档问答助手，任务是基于检索到的资料回答用户问题。

必须遵守：
1. 只根据给定资料回答，不要编造资料中没有的信息。
2. 如果资料中没有答案，直接说“根据当前文档无法确定”。
3. 关键结论后标注引用编号，例如：[1]。
4. 输出使用以下结构：
   结论：
   依据：
   引用来源：
   不确定信息：
{conversation_block}

资料：
{joined_evidence}

问题：
{question}

答案："""
