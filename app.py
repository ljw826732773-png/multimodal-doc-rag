from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.document_loader import load_document
from src.rag_chain import build_llm_answer
from src.text_splitter import split_pages
from src.vector_store import DocumentVectorStore


ROOT_DIR = Path(__file__).parent
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
CHROMA_DIR = ROOT_DIR / "data" / "chroma"


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
st.caption("第 1 版：先跑通 RAG 闭环，支持 TXT / MD / PDF 的解析、切分、向量检索与引用展示。")

store = get_store()

with st.sidebar:
    st.header("知识库")
    st.write(f"当前 chunk 数：**{store.count()}**")
    uploaded_file = st.file_uploader("上传文档", type=["txt", "md", "pdf"])
    chunk_size = st.slider("Chunk 大小", min_value=300, max_value=1500, value=800, step=100)
    overlap = st.slider("Overlap", min_value=0, max_value=300, value=120, step=20)

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

    if st.button("加载示例文档"):
        with st.spinner("正在加载 examples/sample.txt ..."):
            sample_path = ROOT_DIR / "examples" / "sample.txt"
            pages = load_document(sample_path)
            chunks = split_pages(pages, chunk_size=chunk_size, overlap=overlap)
            count = store.add_chunks(chunks)

        st.success(f"完成：写入 {count} 个示例 chunk。")

    if uploaded_file and st.button("解析并建立索引", type="primary"):
        with st.spinner("正在解析文档并写入向量库..."):
            file_path = save_uploaded_file(uploaded_file)
            pages = load_document(file_path)
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

    if st.button("检索并回答", type="primary", disabled=not question.strip()):
        with st.spinner("正在检索相关片段并生成答案..."):
            hits = store.search(question, top_k=top_k)
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
