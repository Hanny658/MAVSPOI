from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from openai import OpenAI

from src.config import Settings


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(parts)
    return str(content)


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            return json.loads(text[start : end + 1])
        raise


def _normalize_port(raw_port: str) -> str:
    text = str(raw_port or "").strip()
    if text.startswith(":"):
        text = text[1:]
    if not text.isdigit():
        return ""
    value = int(text)
    if value < 1 or value > 65535:
        return ""
    return str(value)


def _resolve_base_url(base_url: str, api_port: str) -> str:
    base_url = str(base_url or "").strip()
    port = _normalize_port(api_port)
    if not base_url and not port:
        return ""
    if not base_url and port:
        return f"http://127.0.0.1:{port}/v1"
    if not port:
        return base_url

    parsed = urlsplit(base_url)
    # If a port is already provided in base_url, preserve it.
    if parsed.port is not None:
        return base_url

    if not parsed.scheme or not parsed.netloc:
        # Support shorthand host/path without scheme, e.g. 127.0.0.1/v1
        if "://" not in base_url:
            parsed = urlsplit("http://" + base_url)
            if not parsed.netloc:
                return base_url
        else:
            return base_url

    host = parsed.hostname
    if not host:
        return base_url

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        auth += "@"
    netloc = f"{auth}{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


class OpenAIService:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        self.settings = settings
        resolved_base_url = _resolve_base_url(
            settings.openai_base_url,
            settings.openai_api_port,
        )
        client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**client_kwargs)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.settings.openai_embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = _extract_text_content(response.choices[0].message.content)
        return _safe_json_loads(raw)
