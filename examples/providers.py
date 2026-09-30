"""Pick a model from environment variables. Shared by the quickstart and the example site.

    JIT_PROVIDER   ollama (default) | claude | openai | azure | databricks

    ollama       OLLAMA_MODEL
    claude       ANTHROPIC_API_KEY, CLAUDE_MODEL
    openai       OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL (optional, for any compatible server),
                 OPENAI_DIALECT (optional: openai or simple)
    azure        AZURE_OPENAI_RESOURCE, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_KEY
    databricks   DATABRICKS_HOST, DATABRICKS_ENDPOINT, DATABRICKS_TOKEN
"""
from __future__ import annotations

import os
from typing import Any

from jit_agents import Claude, Ollama, OpenAICompatible

env = os.environ.get


def available() -> dict[str, Any]:
    """Every provider that has enough configuration to be tried."""
    found: dict[str, Any] = {"ollama": Ollama(env("OLLAMA_MODEL", "qwen3.8:27b-mlx"))}
    found["claude"] = Claude(env("CLAUDE_MODEL", "claude-opus-5-5"))
    if env("OPENAI_MODEL"):
        found["openai"] = OpenAICompatible(
            env("OPENAI_MODEL"), base_url=env("OPENAI_BASE_URL"), api_key=env("OPENAI_API_KEY"),
            dialect=env("OPENAI_DIALECT", "openai"),
        )
    if env("AZURE_OPENAI_RESOURCE") and env("AZURE_OPENAI_DEPLOYMENT"):
        found["azure"] = OpenAICompatible.azure(env("AZURE_OPENAI_RESOURCE"), env("AZURE_OPENAI_DEPLOYMENT"))
    if env("DATABRICKS_HOST") and env("DATABRICKS_ENDPOINT"):
        found["databricks"] = OpenAICompatible.databricks(env("DATABRICKS_HOST"), env("DATABRICKS_ENDPOINT"))
    return found


def default_name(providers: dict[str, Any]) -> str:
    chosen = env("JIT_PROVIDER")
    if chosen:
        if chosen not in providers:
            raise SystemExit(f"JIT_PROVIDER={chosen} is not configured. See examples/providers.py for the variables it needs.")
        return chosen
    return "claude" if (env("ANTHROPIC_API_KEY") or env("ANTHROPIC_AUTH_TOKEN")) else "ollama"


def from_env() -> tuple[str, Any]:
    providers = available()
    name = default_name(providers)
    return name, providers[name]
