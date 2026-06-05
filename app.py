from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from src.document_loader import load_document
from src.document_loader import IMAGE_SUFFIXES
from src.document_loader import PageText
from src.evaluation import run_retrieval_eval
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


st.title("智能文档理解与知识问答系统")
st.caption("支持文本 PDF、扫描件 OCR、图片 OCR、RAG 检索、重排、DeepSeek 生成与引用溯源。")

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

tab_qa, tab_eval = st.tabs(["问答", "评测"])

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
        top_k = st.slider("检索 Top-K", min_value=1, max_value=10, value=5)
        fetch_k = st.slider("召回 Fetch-K", min_value=top_k, max_value=20, value=max(8, top_k))
        retrieval_label = st.segmented_control("检索模式", ["向量检索", "混合检索"], default="混合检索")
        use_rerank = st.checkbox("启用轻量重排", value=True)

        if st.button("检索并回答", type="primary", disabled=not question.strip()):
            with st.spinner("正在检索相关片段并生成答案..."):
                if retrieval_label == "混合检索":
                    hits = store.search_hybrid(
                        question,
                        top_k=fetch_k,
                        vector_k=fetch_k,
                        keyword_k=fetch_k,
                    )
                else:
                    hits = store.search(question, top_k=fetch_k)
                if use_rerank:
                    hits = rerank_hits(question, hits, top_k=top_k)
                else:
                    hits = hits[:top_k]
                st.session_state["last_hits"] = hits
                st.session_state["last_answer"] = build_llm_answer(
                    question,
                    hits,
                    provider=llm_provider,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )

        if "last_answer" in st.session_state:
            st.markdown("#### 回答")
            st.write(st.session_state["last_answer"])

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

with tab_eval:
    st.subheader("检索评测")
    st.write("使用示例文档和问题集评测检索片段是否覆盖标准关键词。")
    eval_top_k = st.slider("评测 Top-K", min_value=1, max_value=10, value=3)
    eval_fetch_k = st.slider("评测 Fetch-K", min_value=eval_top_k, max_value=20, value=8)
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
