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

- 接入 OpenAI / 兼容 API 大模型
- 支持 OCR 和扫描 PDF
- 接入 Qwen-VL 或 MiniCPM-V 做图片、图表、截图理解
- 增加 Rerank 和 Hybrid Search
- 增加 RAG 评测模块

## 免费本地大模型

当前版本支持 Ollama 本地模型。Ollama 免费运行在你的电脑上，不需要 API Key。

安装 Ollama 后，下载一个小模型：

```bash
ollama pull qwen2.5:3b
```

启动应用后，在侧边栏把 `LLM Provider` 选择为 `Ollama 本地免费模型`，模型名保持：

```text
qwen2.5:3b
```

如果没有安装或没有启动 Ollama，系统会自动降级为检索结果草稿。

## DeepSeek API

当前版本也支持 DeepSeek API。启动应用后，在侧边栏选择：

```text
LLM Provider: DeepSeek API
DeepSeek 模型名: deepseek-v4-flash
DeepSeek 地址: https://api.deepseek.com
```

然后在 `DeepSeek API Key` 输入框中填入你的 key。Key 只在当前页面会话里使用，不会写入代码或提交到 GitHub。

也可以用环境变量设置：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
streamlit run app.py
```
