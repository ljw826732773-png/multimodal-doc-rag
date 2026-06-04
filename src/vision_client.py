from __future__ import annotations

import base64
import json
from pathlib import Path
import urllib.error
import urllib.request


class VisionError(RuntimeError):
    pass


def describe_image_openai_compatible(
    image_path: str | Path,
    api_key: str,
    model: str,
    base_url: str,
    prompt: str = "请用中文描述这张图片中的关键信息；如果有图表，请总结趋势、数值和结论。",
    timeout: int = 120,
) -> str:
    if not api_key.strip():
        raise VisionError("请先填写视觉模型 API Key。")

    image_path = Path(image_path)
    mime_type = _mime_type(image_path)
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                    },
                ],
            }
        ],
        "temperature": 0.2,
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise VisionError(f"视觉模型请求失败：HTTP {exc.code}，{error_body}") from exc
    except urllib.error.URLError as exc:
        raise VisionError(f"无法连接视觉模型接口：{exc.reason}") from exc

    result = json.loads(body)
    try:
        description = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionError(f"视觉模型返回格式异常：{result}") from exc

    if not description:
        raise VisionError("视觉模型返回了空描述。")
    return description


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")
