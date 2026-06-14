from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.answer_diagnostics import build_answer_diagnostics
from src.conversation import build_contextual_query, build_conversation_context
from src.document_loader import IMAGE_SUFFIXES, PageText, load_document
from src.evaluation import run_retrieval_eval
from src.query_router import route_query
from src.rag_chain import build_llm_answer
from src.reranker import rerank_hits
from src.text_splitter import split_pages
from src.vector_store import DocumentVectorStore
from src.vision_client import VisionError, describe_image_openai_compatible


ROOT_DIR = Path(__file__).parent
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
CHROMA_DIR = ROOT_DIR / "data" / "chroma"
EVAL_CHROMA_DIR = ROOT_DIR / "data" / "eval_chroma"


st.set_page_config(page_title="Multimodal Doc RAG", page_icon="doc", layout="wide")


def save_uploaded_file(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


@st.cache_resource
def get_store() -> DocumentVectorStore:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return DocumentVectorStore(CHROMA_DIR)


def add_history(
    question: str,
    answer: str,
    hits: list[dict],
    diagnostics: dict | None = None,
    retrieval_query: str | None = None,
) -> None:
    st.session_state.setdefault("qa_history", [])
    st.session_state["qa_history"].append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "retrieval_query": retrieval_query or question,
            "answer": answer,
            "citations": [
                {
                    "source": hit.get("source"),
                    "page": hit.get("page"),
                    "chunk_index": hit.get("chunk_index"),
                    "score": round(float(hit.get("score", 0.0)), 4),
                }
                for hit in hits
            ],
            "diagnostics": diagnostics or {},
        }
    )


st.title("智能文档理解与知识问答系统")
st.caption("支持文档解析、OCR、图片语义描述、Hybrid Search、重排、DeepSeek 生成与引用溯源。")

store = get_store()

with st.sidebar:
    st.header("知识库")
    st.write(f"当前 chunk 数：**{store.count()}**")
    uploaded_file = st.file_uploader(
        "上传文档或图片",
        type=["txt", "md", "pdf", "png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
    )
    enable_ocr = st.checkbox("启用 OCR（图片/扫描 PDF）", value=True)
    chunk_size = st.slider("Chunk 大小", min_value=300, max_value=1500, value=800, step=100)
    overlap = st.slider("Overlap", min_value=0, max_value=300, value=120, step=20)

    if st.button("清空知识库"):
        deleted = store.clear()
        st.session_state.pop("last_hits", None)
        st.session_state.pop("last_answer", None)
        st.session_state.pop("last_diagnostics", None)
        st.success(f"已清空 {deleted} 个 chunk。")

    st.divider()
    st.header("模型设置")
    provider_label = st.selectbox(
        "LLM Provider",
        ["DeepSeek API", "关闭生成，只看检索", "Ollama 本地免费模型"],
    )
    if provider_label == "DeepSeek API":
        llm_provider = "deepseek"
        model = st.text_input("DeepSeek 模型名", value="deepseek-v4-flash")
        base_url = st.text_input("DeepSeek 地址", value="https://api.deepseek.com")
        api_key = st.text_input(
            "DeepSeek API Key",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            type="password",
            help="只在当前页面会话里使用，不会写入代码或 GitHub。",
        )
    elif provider_label == "Ollama 本地免费模型":
        llm_provider = "ollama"
        model = st.text_input("Ollama 模型名", value="qwen2.5:3b")
        base_url = st.text_input("Ollama 地址", value="http://localhost:11434")
        api_key = ""
    else:
        llm_provider = "disabled"
        model = "disabled"
        base_url = ""
        api_key = ""

    st.divider()
    st.header("视觉理解")
    enable_vision = st.checkbox("启用视觉模型描述（图片/图表）", value=False)
    vision_model = st.text_input("视觉模型名", value="qwen-vl-max")
    vision_base_url = st.text_input("视觉模型地址", value="")
    vision_api_key = st.text_input(
        "视觉模型 API Key",
        value=os.getenv("VISION_API_KEY", ""),
        type="password",
        help="可填写任意 OpenAI-compatible 视觉模型接口；DeepSeek 当前主要用于文本问答。",
    )

tab_qa, tab_kb, tab_history, tab_eval = st.tabs(["问答", "知识库管理", "问答历史", "评测"])

with tab_qa:
    if st.button("加载示例文档"):
        with st.spinner("正在加载 examples/sample.txt ..."):
            sample_path = ROOT_DIR / "examples" / "sample.txt"
            pages = load_document(sample_path, enable_ocr=False)
            chunks = split_pages(pages, chunk_size=chunk_size, overlap=overlap)
            count = store.add_chunks(chunks)

        st.success(f"完成：写入 {count} 个示例 chunk。")

    if uploaded_file and st.button("解析并建立索引", type="primary"):
        with st.spinner("正在解析文档并写入向量库..."):
            file_path = save_uploaded_file(uploaded_file)
            pages = load_document(file_path, enable_ocr=enable_ocr)
            if enable_vision and file_path.suffix.lower() in IMAGE_SUFFIXES:
                try:
                    description = describe_image_openai_compatible(
                        image_path=file_path,
                        api_key=vision_api_key,
                        model=vision_model,
                        base_url=vision_base_url,
                    )
                    pages.append(
                        PageText(
                            source=file_path.name,
                            page=1,
                            text=f"视觉模型描述：\n{description}",
                        )
                    )
                except VisionError as exc:
                    st.warning(str(exc))
            chunks = split_pages(pages, chunk_size=chunk_size, overlap=overlap)
            count = store.add_chunks(chunks)

        st.success(f"完成：解析 {len(pages)} 页，写入 {count} 个 chunk。")

    left, right = st.columns([0.58, 0.42], gap="large")

    with left:
        st.subheader("提问")
        question = st.text_area(
            "输入你想问文档的问题",
            height=120,
            placeholder="例如：这篇文档主要解决什么问题？",
        )
        auto_route = st.checkbox("自动选择检索策略", value=True)
        use_conversation = st.checkbox("启用多轮上下文", value=True)
        history_turns = st.slider("上下文轮数", min_value=1, max_value=5, value=3)
        top_k = st.slider("检索 Top-K", min_value=1, max_value=10, value=5)
        fetch_k = st.slider("召回 Fetch-K", min_value=top_k, max_value=20, value=max(8, top_k))
        retrieval_label = st.segmented_control("检索模式", ["向量检索", "混合检索"], default="混合检索")
        use_rerank = st.checkbox("启用轻量重排", value=True)

        if st.button("检索并回答", type="primary", disabled=not question.strip()):
            with st.spinner("正在检索相关片段并生成答案..."):
                history = st.session_state.get("qa_history", [])
                conversation_context = (
                    build_conversation_context(history, max_turns=history_turns)
                    if use_conversation
                    else ""
                )
                retrieval_query = (
                    build_contextual_query(question, history, max_turns=history_turns)
                    if use_conversation
                    else question
                )
                if use_conversation and conversation_context:
                    st.info("已使用最近问答作为多轮上下文增强检索。")

                route = route_query(retrieval_query) if auto_route else None
                effective_mode = route.retrieval_mode if route else ("hybrid" if retrieval_label == "混合检索" else "vector")
                effective_top_k = route.top_k if route else top_k
                effective_fetch_k = route.fetch_k if route else fetch_k
                effective_rerank = route.use_rerank if route else use_rerank
                if route:
                    st.info(
                        f"自动策略：{route.intent} | {route.retrieval_mode} | "
                        f"Top-K={route.top_k}, Fetch-K={route.fetch_k}。{route.reason}"
                    )

                if effective_mode == "hybrid":
                    hits = store.search_hybrid(
                        retrieval_query,
                        top_k=effective_fetch_k,
                        vector_k=effective_fetch_k,
                        keyword_k=effective_fetch_k,
                    )
                else:
                    hits = store.search(retrieval_query, top_k=effective_fetch_k)
                if effective_rerank:
                    hits = rerank_hits(retrieval_query, hits, top_k=effective_top_k)
                else:
                    hits = hits[:effective_top_k]
                answer = build_llm_answer(
                    question,
                    hits,
                    provider=llm_provider,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    conversation_context=conversation_context,
                )
                diagnostics = build_answer_diagnostics(question, answer, hits)
                st.session_state["last_hits"] = hits
                st.session_state["last_answer"] = answer
                st.session_state["last_diagnostics"] = diagnostics
                st.session_state["last_retrieval_query"] = retrieval_query
                add_history(question, answer, hits, diagnostics, retrieval_query)

        if "last_answer" in st.session_state:
            st.markdown("#### 回答")
            st.write(st.session_state["last_answer"])
            retrieval_query = st.session_state.get("last_retrieval_query")
            if retrieval_query and retrieval_query != question:
                with st.expander("本轮实际用于检索的上下文查询", expanded=False):
                    st.code(retrieval_query)

    with right:
        st.subheader("引用来源")
        hits = st.session_state.get("last_hits", [])
        if not hits:
            st.info("提问后这里会显示检索命中的原文片段。")
        for hit in hits:
            with st.expander(
                f"{hit['source']} | 第 {hit['page']} 页 | chunk {hit['chunk_index']} | score {hit['score']:.3f}",
                expanded=False,
            ):
                st.write(hit["text"])

        diagnostics = st.session_state.get("last_diagnostics")
        if diagnostics:
            st.divider()
            st.subheader("答案诊断")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("可信度", diagnostics.get("confidence", "unknown"))
            col_b.metric("证据覆盖", diagnostics.get("answer_coverage", 0.0))
            col_c.metric("来源数", diagnostics.get("source_count", 0))
            st.progress(float(diagnostics.get("confidence_score", 0.0)))
            warnings = diagnostics.get("warnings", [])
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success("当前答案与检索证据匹配较好。")
            st.dataframe(diagnostics.get("evidence_rows", []), use_container_width=True)

with tab_kb:
    st.subheader("已索引文档")
    documents = store.list_documents()
    if not documents:
        st.info("当前知识库为空。")
    else:
        st.dataframe(documents, use_container_width=True)
        selected_file_id = st.selectbox(
            "选择要删除的文档",
            [doc["file_id"] for doc in documents],
            format_func=lambda value: next(
                (doc["source"] for doc in documents if doc["file_id"] == value),
                value,
            ),
        )
        if st.button("删除所选文档", type="primary"):
            deleted = store.delete_document(selected_file_id)
            st.success(f"已删除 {deleted} 个 chunk。")
            st.rerun()

with tab_history:
    st.subheader("问答历史")
    history = st.session_state.get("qa_history", [])
    if not history:
        st.info("当前还没有问答记录。")
    else:
        for item in reversed(history):
            with st.expander(f"{item['time']} | {item['question']}", expanded=False):
                st.write(item["answer"])
                if item.get("diagnostics"):
                    st.write("诊断：")
                    st.json(item["diagnostics"])
                if item.get("retrieval_query") and item["retrieval_query"] != item["question"]:
                    st.write("检索查询：")
                    st.code(item["retrieval_query"])
                st.write("引用：")
                st.json(item["citations"])

        history_json = json.dumps(history, ensure_ascii=False, indent=2)
        st.download_button(
            "导出问答历史 JSON",
            data=history_json,
            file_name="qa_history.json",
            mime="application/json",
        )
        if st.button("清空问答历史"):
            st.session_state["qa_history"] = []
            st.rerun()

with tab_eval:
    st.subheader("检索评测")
    st.write("使用示例文档和问题集评测检索片段是否覆盖标准关键词。")
    eval_top_k = st.slider("评测 Top-K", min_value=1, max_value=10, value=3)
    eval_fetch_k = st.slider("评测 Fetch-K", min_value=eval_top_k, max_value=20, value=8)
    eval_auto_route = st.checkbox("评测启用自动策略", value=True)
    eval_retrieval_label = st.segmented_control("评测检索模式", ["向量检索", "混合检索"], default="混合检索")
    eval_rerank = st.checkbox("评测启用轻量重排", value=True)

    if st.button("运行示例评测", type="primary"):
        with st.spinner("正在运行评测..."):
            report = run_retrieval_eval(
                document_path=ROOT_DIR / "examples" / "sample.txt",
                questions_path=ROOT_DIR / "eval" / "questions.json",
                persist_dir=EVAL_CHROMA_DIR,
                top_k=eval_top_k,
                fetch_k=eval_fetch_k,
                use_rerank=eval_rerank,
                retrieval_mode="hybrid" if eval_retrieval_label == "混合检索" else "vector",
                use_router=eval_auto_route,
            )
        st.metric("关键词召回率", report["keyword_recall"])
        st.metric("耗时（秒）", report["elapsed_seconds"])
        st.dataframe(report["rows"], use_container_width=True)
        st.download_button(
            "下载评测报告 JSON",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="rag_eval_report.json",
            mime="application/json",
        )
