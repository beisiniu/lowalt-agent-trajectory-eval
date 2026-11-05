"""OpenAI 兼容的极简 LLM 客户端（零第三方依赖）。

为什么不用 openai SDK
--------------------
本机 pypi.org 直连超时，只配了国内镜像；少一个依赖少一处坑。
llama.cpp 的 `llama-server` 与 vLLM / Ollama / 商用 API 都兼容
`POST /v1/chat/completions`，用标准库 urllib 足够。

端点选择优先级
--------------
1. 显式传参 base_url
2. 环境变量 LOWALT_LLM_BASE
3. 默认 http://127.0.0.1:8081/v1 （llama-server 默认端口）

模型名同理：LOWALT_LLM_MODEL，默认取 /v1/models 返回的第一个。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

# 本机推理服务必须绕过系统代理。
# 若环境里配了 http_proxy，urllib 会把 127.0.0.1 的请求也发给代理，
# 表现是返回 502/404 而不是连不上——很难定位。这里显式豁免。
_LOCAL_HOSTS = "127.0.0.1,localhost,::1"
for _k in ("NO_PROXY", "no_proxy"):
    _v = os.environ.get(_k, "")
    if "127.0.0.1" not in _v:
        os.environ[_k] = f"{_v},{_LOCAL_HOSTS}".strip(",")

# 无代理 opener，用于所有本地端点请求
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 2,
    ):
        self.base_url = (base_url or os.getenv("LOWALT_LLM_BASE") or "http://127.0.0.1:8081/v1").rstrip("/")
        self.model = model or os.getenv("LOWALT_LLM_MODEL") or ""
        self.api_key = api_key or os.getenv("LOWALT_LLM_KEY") or "not-needed"
        self.timeout = timeout
        self.max_retries = max_retries
        self._resolved_model: str | None = None

    # ---------------- 基础能力 ----------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with _NO_PROXY_OPENER.open(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise LLMError(f"请求 {url} 失败: {e}") from e
        raise LLMError(f"请求 {url} 失败: {last_err}")

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        if self._resolved_model:
            return self._resolved_model
        url = f"{self.base_url}/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with _NO_PROXY_OPENER.open(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("data") or []
            if items:
                self._resolved_model = items[0].get("id", "")
                return self._resolved_model
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"无法从 {url} 解析模型名: {e}") from e
        raise LLMError(f"{url} 未返回任何模型")

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        data = self._post("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"响应格式异常: {json.dumps(data, ensure_ascii=False)[:300]}") from e

    # ---------------- 健康检查 ----------------

    def health(self) -> dict[str, Any]:
        try:
            model = self.resolve_model()
            t0 = time.time()
            txt = self.chat([{"role": "user", "content": "回复两个字：就绪"}], max_tokens=16)
            return {
                "ok": True,
                "base_url": self.base_url,
                "model": model,
                "latency_s": round(time.time() - t0, 2),
                "sample": txt.strip()[:40],
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "base_url": self.base_url, "error": str(e)}
