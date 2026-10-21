"""Async OpenAI-compatible client for vLLM / LMDeploy / SGLang servers."""
from __future__ import annotations

from typing import Dict, List, Optional

import httpx


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 8.0, max_connections: int = 256):
        self.base_url, self.model, self.timeout_s = base_url.rstrip("/"), model, timeout_s
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=2.0),
                                      limits=httpx.Limits(max_connections=max_connections,
                                                          max_keepalive_connections=max_connections))

    async def chat(self, messages: List[Dict], max_tokens: int = 256, temperature: float = 0.3,
                   timeout_s: Optional[float] = None) -> Dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = await self.http.post(f"{self.base_url}/chat/completions", json=body,
                                     timeout=timeout_s or self.timeout_s)
        except httpx.TimeoutException as e:
            raise LLMError(f"timeout>{timeout_s or self.timeout_s}s") from e
        except httpx.HTTPError as e:
            raise LLMError(f"conn_error:{type(e).__name__}") from e
        if r.status_code != 200:
            raise LLMError(f"http_{r.status_code}:{r.text[:120]}")
        j = r.json()
        u = j.get("usage") or {}
        return {"text": j["choices"][0]["message"]["content"] or "", "finish_reason": j["choices"][0].get("finish_reason"),
                "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0)}

    def stream(self, body: Dict):
        """Return an async context manager streaming raw SSE bytes (for passthrough)."""
        body = {**body, "model": body.get("model") or self.model}
        return self.http.stream("POST", f"{self.base_url}/chat/completions", json=body)

    async def close(self):
        await self.http.aclose()
