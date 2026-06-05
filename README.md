# Multimodal Doc RAG

基于大模型的智能文档理解与知识问答系统。项目支持文档上传、OCR、多模态图片描述、RAG 检索、轻量重排、DeepSeek 生成答案和引用溯源。

## 当前功能

- 上传 TXT / MD / PDF 文档
- 上传 PNG / JPG / JPEG / BMP / WEBP / TIF / TIFF 图片
- 提取文档文本、页码和来源信息
- 使用 OCR 解析图片和扫描 PDF 页面
- 可选接入 OpenAI-compatible 视觉模型，为图片和图表生成语义描述
- 将长文本切分为 chunk
- 使用 `BAAI/bge-small-zh-v1.5` 生成 embedding
- 使用 ChromaDB 本地向量库做语义检索
- 支持向量检索和 Hybrid Search
- 支持轻量 rerank，对召回片段进行二次排序
- 支持 DeepSeek API 生成结构化答案
- 展示答案引用来源
- 页面内置示例评测面板
- 支持清空知识库，便于对比不同文档和参数

## 快速启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

启动后打开：

```text
http://localhost:8501
```

如果暂时没有文档，可以先点击左侧的“加载示例文档”，然后提问：

```text
What does RAG do when a user asks a question?
```

## 项目流程

```text
Document / Image
-> Text Extraction / OCR / Vision Description
-> Chunking
-> Embedding
-> Vector Store
-> Retrieval
-> Rerank
-> LLM Generation
-> Citation
```

## DeepSeek API

启动应用后，在侧边栏选择：

```text
LLM Provider: DeepSeek API
DeepSeek 模型名: deepseek-v4-flash
DeepSeek 地址: https://api.deepseek.com
```

然后在 `DeepSeek API Key` 输入框中填入你的 key。Key 只在当前页面会话中使用，不会写入代码或提交到 GitHub。

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

## 视觉模型描述

DeepSeek API 当前主要用于文本问答。本项目把图片理解做成可插拔的 OpenAI-compatible 视觉接口。

可在侧边栏开启：

```text
启用视觉模型描述（图片/图表）
视觉模型名
视觉模型地址
视觉模型 API Key
```

开启后，上传图片时系统会同时执行 OCR 和视觉模型描述，把“图像语义描述”也写入知识库。这样不仅能问图片上的文字，还能问图表趋势、画面内容和视觉结论。

## 检索与重排

页面中有两个检索参数：

```text
Top-K: 最终交给模型的片段数量
Fetch-K: 先从向量库召回的候选片段数量
```

检索模式支持：

```text
向量检索: 使用 embedding 相似度召回片段
混合检索: 合并向量相似度和关键词匹配结果
```

启用轻量重排后，系统会先召回更多片段，再结合向量相似度和关键词重合度做二次排序。当前 rerank 是轻量实现，后续可以替换为 `bge-reranker` 这类专门的重排模型。

## 评测

可以在页面的 `评测` 标签页直接运行示例评测，也可以用命令行：

```bash
python scripts/run_eval.py --rerank
```

也可以评测混合检索：

```bash
python scripts/run_eval.py --rerank --retrieval-mode hybrid
```

评测脚本会输出：

- 问题数
- Top-K / Fetch-K
- 是否启用 rerank
- 关键词召回率
- 总耗时
- 每个问题命中的 top source 和相似度

## 技术栈

- Python
- Streamlit
- PyMuPDF
- RapidOCR
- Sentence Transformers
- ChromaDB
- DeepSeek API

## 后续计划

- 接入更强的视觉语言模型，例如 Qwen-VL / MiniCPM-V
- 使用专业 reranker 模型替换轻量重排
- 增加 Hybrid Search
- 增加更完整的 RAG 评测集
- 增加 Docker 部署
- 整理项目截图和简历描述
