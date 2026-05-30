from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def generate_with_ollama(
    prompt: str,
    model: str = "qwen2.5:3b",
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    endpoint = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    body = _post_json(endpoint, payload, timeout=timeout)
    answer = body.get("response", "").strip()
    if not answer:
        raise LLMError("Ollama 返回了空答案。")
    return answer


def generate_with_deepseek(
    prompt: str,
    api_key: str,
    model: str = "deepseek-v4-flash",
    base_url: str = "https://api.deepseek.com",
    timeout: int = 120,
) -> str:
    if not api_key.strip():
        raise LLMError("请先填写 DeepSeek API Key。")

    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严谨的文档问答助手，只根据用户提供的资料回答。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    body = _post_json(
        endpoint,
        payload,
        headers={"Authorization": f"Bearer {api_key.strip()}"},
        timeout=timeout,
    )

    try:
        answer = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"DeepSeek 返回格式异常：{body}") from exc

    if not answer:
        raise LLMError("DeepSeek 返回了空答案。")
    return answer


def _post_json(
    endpoint: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers=request_headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise LLMError(f"模型接口请求失败：HTTP {exc.code}，{error_body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"无法连接模型接口：{exc.reason}") from exc

    return json.loads(raw_body)
