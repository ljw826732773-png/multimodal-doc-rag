# 使用说明

## 1. 启动项目

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开：

```text
http://localhost:8501
```

## 2. 配置 DeepSeek

推荐在页面侧边栏填写：

```text
LLM Provider: DeepSeek API
DeepSeek 模型名: deepseek-v4-flash
DeepSeek 地址: https://api.deepseek.com
DeepSeek API Key: 你的 key
```

也可以用环境变量：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
streamlit run app.py
```

不要把 API Key 写进代码，也不要提交到 GitHub。

## 3. 上传文档

支持格式：

```text
TXT / MD / PDF
PNG / JPG / JPEG / BMP / WEBP / TIF / TIFF
```

普通 PDF 会先尝试直接提取文本。扫描 PDF 或图片会走 OCR。

## 4. 建立索引

上传文件后点击：

```text
解析并建立索引
```

系统会执行：

```text
解析文本 -> 切分 chunk -> 生成 embedding -> 写入 ChromaDB
```

## 5. 提问

在 `问答` 标签页输入问题，调整：

```text
自动选择检索策略
Top-K
Fetch-K
检索模式
是否启用轻量重排
```

开启 `自动选择检索策略` 后，系统会根据问题自动选择检索模式、Top-K、Fetch-K 和 rerank 设置。关闭后，可以手动选择向量检索或混合检索。

然后点击：

```text
检索并回答
```

右侧会显示引用来源。

## 6. 清空知识库

侧边栏的 `清空知识库` 可以删除当前向量库中的所有 chunk。测试不同文档前建议先清空，避免旧文档影响结果。

## 7. 管理知识库

`知识库管理` 标签页会显示当前已索引文档，包括：

```text
file_id
source
chunks
pages
```

可以选择某个文档并删除它对应的全部 chunk。

## 8. 查看问答历史

`问答历史` 标签页会保存当前页面会话中的问题、答案和引用来源。可以导出为 JSON，也可以清空历史。

## 9. 运行评测

页面方式：

```text
评测 -> 运行示例评测
```

命令行方式：

```bash
python scripts/run_eval.py --rerank
python scripts/run_eval.py --rerank --retrieval-mode hybrid
python scripts/run_eval.py --router
```

评测结果会输出关键词召回率、耗时和每个问题命中的 top source。
