"""Model providers. Anything with an async ``json`` method works."""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Protocol, runtime_checkable

import httpx


class ModelError(Exception):
    pass


@runtime_checkable
class Model(Protocol):
    async def json(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON object that follows ``schema``."""
        ...


class Ollama:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout_s: float = 600.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s

    async def json(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "format": schema,
            "think": False,
            "options": {"temperature": self.temperature},
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            try:
                resp = await client.post(f"{self.host}/api/chat", json=body)
            except httpx.HTTPError as exc:
                raise ModelError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        if resp.status_code != 200:
            raise ModelError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")
        content = resp.json().get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelError(f"model returned non-JSON: {content[:300]}") from exc


class Claude:
    """Claude through the official Anthropic SDK. Install with the ``claude`` extra.

    Credentials come from the environment (``ANTHROPIC_API_KEY``, or a profile
    from ``ant auth login``). Responses use structured outputs, so they always
    follow the schema.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str | None = None,
        max_tokens: int = 16000,
        fallbacks: bool = True,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ModelError("the anthropic package is not installed; install jit-agents[claude]") from exc
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def json(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        import anthropic

        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)}
        }
        if self.effort:
            output_config["effort"] = self.effort
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "output_config": output_config,
        }
        try:
            if self.fallbacks:
                # If a safety classifier declines, the API re-runs the request on a fallback model.
                response = await self.client.beta.messages.create(
                    betas=[self.FALLBACK_BETA], fallbacks="default", **request
                )
            else:
                response = await self.client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise ModelError("Claude rejected the credentials; check ANTHROPIC_API_KEY") from exc
        except anthropic.RateLimitError as exc:
            raise ModelError("Claude rate limit reached; try again shortly") from exc
        except anthropic.APIStatusError as exc:
            raise ModelError(f"Claude returned {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ModelError(f"cannot reach the Claude API: {exc}") from exc
        except (anthropic.AnthropicError, TypeError) as exc:
            if "authentication" in str(exc).lower():
                raise ModelError(
                    "no Claude credentials found; set ANTHROPIC_API_KEY in the environment that starts this process"
                ) from exc
            raise ModelError(str(exc)) from exc

        if response.stop_reason == "refusal":
            raise ModelError("Claude declined this request")
        if response.stop_reason == "max_tokens":
            raise ModelError(f"Claude ran out of output tokens ({self.max_tokens}); raise max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelError(f"Claude returned non-JSON: {text[:300]}") from exc


class CallableModel:
    """Wrap a function as a model. Useful for tests and custom providers."""

    def __init__(self, fn: Callable[[str, list[dict[str, str]], dict[str, Any]], Any]) -> None:
        self.fn = fn

    async def json(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        result = self.fn(system, messages, schema)
        if inspect.isawaitable(result):
            result = await result
        return result
