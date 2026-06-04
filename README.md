# Multimodal Doc RAG

基于多模态大模型的智能文档理解与知识问答系统。当前是第 1 版：先跑通 RAG 最小闭环。

## 当前功能

- 上传 TXT / MD / PDF
- 上传 PNG / JPG / JPEG / BMP / WEBP / TIF / TIFF 图片
- 提取文档文本和页码
- 使用 OCR 解析图片和扫描 PDF 页面
- 将长文本切分为 chunk
- 使用 `BAAI/bge-small-zh-v1.5` 生成 embedding
- 使用 ChromaDB 本地向量库检索
- 支持 DeepSeek API 生成结构化答案
- 支持轻量 rerank，对召回片段进行二次排序
- 展示回答和引用来源
- 提供基础评测脚本
- 页面内置示例评测面板
- 支持清空知识库，便于对比不同文档和参数

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

## OCR 和图片

项目使用 `rapidocr-onnxruntime` 做本地 OCR，支持：

```text
图片 OCR
扫描 PDF 的空文本页面 OCR
```

上传图片或扫描件时，保持侧边栏的 `启用 OCR（图片/扫描 PDF）` 勾选即可。OCR 会把识别出的文字作为文档内容写入向量库，后续可以继续用 RAG 提问。

## 检索与重排

页面中有两个检索参数：

```text
Top-K: 最终交给模型的片段数量
Fetch-K: 先从向量库召回的候选片段数量
```

启用轻量重排后，系统会先召回更多片段，再结合向量相似度和关键词重合度做二次排序。这个版本的 rerank 是轻量实现，后续可以替换为 `bge-reranker` 这类专门的重排模型。

## 评测

可以在页面的 `评测` 标签页直接运行示例评测，也可以用命令行：

运行基础评测：

```bash
python scripts/run_eval.py --rerank
```

评测脚本会输出问题数、关键词召回率、耗时和每个问题命中的 top source。后续可以扩展为更完整的准确率、召回率和延迟对比实验。
