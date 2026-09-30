"""Models behind an OpenAI-format endpoint: OpenAI, Azure Foundry, Databricks, and local servers.

Install with the ``openai`` extra. The model must support structured outputs.
There is no fallback to free-form text, because the library depends on replies
that follow a schema.
"""
from __future__ import annotations

import json
import os
from typing import Any, Literal

from jit_agents.llm import ModelError
from jit_agents.schema import decode, to_simple, to_strict

Dialect = Literal["openai", "simple"]


def _text(content: Any) -> str:
    """Reply text. Some hosts return a list of blocks, with reasoning kept apart from the answer."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if kind in ("text", "output_text"):
                parts.append(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", ""))
        return "".join(parts)
    return ""


class OpenAICompatible:
    """Any model served in the OpenAI chat completions format.

    ``dialect`` picks how schemas are adapted:

    - ``"openai"``: OpenAI and Azure Foundry. The full schema is sent in strict mode.
    - ``"simple"``: Databricks. No ``anyOf`` or ``$ref`` and a cap on keys, so parts of
      the schema are sent as JSON held in strings and decoded afterwards.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: Any | None = None,
        dialect: Dialect = "openai",
        max_schema_keys: int = 64,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        retries: int = 2,
        timeout_s: float = 600.0,
        client: Any | None = None,
    ) -> None:
        if dialect not in ("openai", "simple"):
            raise ValueError("dialect must be 'openai' or 'simple'")
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.dialect = dialect
        self.max_schema_keys = max_schema_keys
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body
        self.retries = retries
        self.timeout_s = timeout_s
        self._client = client

    # -- ready-made hosts -----------------------------------------------------
    @classmethod
    def azure(cls, resource: str, deployment: str, api_key: Any | None = None, **kwargs: Any) -> "OpenAICompatible":
        """Azure Foundry. ``deployment`` is your model deployment name.

        ``api_key`` may be a key or an async token provider for Microsoft Entra ID.
        Defaults to the ``AZURE_OPENAI_API_KEY`` environment variable.
        """
        return cls(
            deployment, base_url=f"https://{resource}.openai.azure.com/openai/v1/",
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"), dialect="openai", **kwargs,
        )

    @classmethod
    def databricks(cls, host: str, endpoint: str, token: str | None = None, **kwargs: Any) -> "OpenAICompatible":
        """Databricks model serving. ``host`` is the workspace URL and ``endpoint`` the serving endpoint name.

        Defaults to the ``DATABRICKS_TOKEN`` environment variable.
        """
        if not host.startswith("http"):
            host = "https://" + host
        return cls(
            endpoint, base_url=f"{host.rstrip('/')}/serving-endpoints",
            api_key=token or os.environ.get("DATABRICKS_TOKEN"), dialect="simple", **kwargs,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ModelError("the openai package is not installed; install jit-agents[openai]") from exc
            try:
                self._client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)
            except openai.OpenAIError as exc:
                raise ModelError(f"no credentials for {self.base_url or 'OpenAI'}: pass api_key or set its environment variable") from exc
        return self._client

    def prepare(self, schema: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        """The schema as it will be sent, and the plan for decoding the reply."""
        if self.dialect == "simple":
            return to_simple(schema, self.max_schema_keys)
        return to_strict(schema), None

    async def json(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        client = self.client
        import openai

        sent, plan = self.prepare(schema)
        request: dict[str, Any] = {
            "model": self.model,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "jit_agents_response", "schema": sent, "strict": True},
            },
        }
        if self.max_tokens is not None:
            request["max_completion_tokens"] = self.max_tokens
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            request["extra_body"] = self.extra_body

        conversation = [{"role": "system", "content": system}, *[{"role": m["role"], "content": m["content"]} for m in messages]]
        problem = "no reply"
        for _ in range(1 + self.retries):
            try:
                response = await client.chat.completions.create(messages=conversation, **request)
            except openai.AuthenticationError as exc:
                raise ModelError(f"{self.model}: the endpoint rejected the credentials") from exc
            except openai.NotFoundError as exc:
                raise ModelError(f"{self.model}: no such model or endpoint at {self.base_url or 'OpenAI'}") from exc
            except openai.RateLimitError as exc:
                raise ModelError(f"{self.model}: rate limit reached; try again shortly") from exc
            except openai.BadRequestError as exc:
                hint = (
                    " The model must support structured outputs."
                    + (" If the host rejects anyOf or $ref, use dialect='simple'." if self.dialect == "openai" else "")
                )
                raise ModelError(f"{self.model}: request rejected: {exc.message}.{hint}") from exc
            except openai.APIStatusError as exc:
                raise ModelError(f"{self.model}: endpoint returned {exc.status_code}: {exc.message}") from exc
            except openai.APIConnectionError as exc:
                raise ModelError(f"{self.model}: cannot reach {self.base_url or 'OpenAI'}: {exc}") from exc

            if not hasattr(response, "choices"):
                # A wrong host or path often answers with a web page instead of an API error.
                raise ModelError(
                    f"{self.model}: {self.base_url or 'the endpoint'} did not answer in the OpenAI format. "
                    f"Check the base URL. It sent: {str(response)[:120]!r}"
                )
            if not response.choices:
                raise ModelError(f"{self.model}: the endpoint returned no choices")
            choice = response.choices[0]
            if getattr(choice.message, "refusal", None):
                raise ModelError(f"{self.model} declined this request: {choice.message.refusal}")
            if choice.finish_reason == "length":
                raise ModelError(f"{self.model} ran out of output tokens; raise max_tokens")
            text = _text(choice.message.content)
            try:
                data = json.loads(text)
                if not isinstance(data, dict):
                    raise ValueError("the reply was not a JSON object")
                return decode(data, plan)
            except ValueError as exc:
                problem = str(exc)
                conversation = conversation + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That reply could not be read: {problem}. Send the complete reply again as valid JSON."},
                ]
        raise ModelError(f"{self.model}: unreadable reply after {1 + self.retries} attempts: {problem}")
