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
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise LLMError(
            "无法连接 Ollama。请确认已安装 Ollama，并且本地服务正在运行。"
        ) from exc

    result = json.loads(body)
    answer = result.get("response", "").strip()
    if not answer:
        raise LLMError("Ollama 返回了空答案。")
    return answer
