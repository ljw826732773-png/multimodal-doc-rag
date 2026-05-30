# Multimodal Doc RAG

基于多模态大模型的智能文档理解与知识问答系统。当前是第 1 版：先跑通 RAG 最小闭环。

## 当前功能

- 上传 TXT / MD / PDF
- 提取文档文本和页码
- 将长文本切分为 chunk
- 使用 `BAAI/bge-small-zh-v1.5` 生成 embedding
- 使用 ChromaDB 本地向量库检索
- 展示回答草稿和引用来源

## 快速启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

如果你暂时没有文档可上传，可以先点击左侧的“加载示例文档”，然后提问：

```text
What does RAG do when a user asks a question?
```

## 第一版学习重点

RAG 的核心流程是：

```text
Document -> Chunk -> Embedding -> Vector Store -> Retrieve -> Generate
```

当前版本已经完成前五步，并用检索结果生成一个“回答草稿”。下一步会接入真正的 LLM，让模型基于检索片段生成自然语言答案。

## 后续计划

- 接入 OpenAI / Ollama / 兼容 API 大模型
- 支持 OCR 和扫描 PDF
- 接入 Qwen-VL 或 MiniCPM-V 做图片、图表、截图理解
- 增加 Rerank 和 Hybrid Search
- 增加 RAG 评测模块
