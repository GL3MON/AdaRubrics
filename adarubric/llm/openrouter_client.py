"""OpenRouter backend using its OpenAI-compatible chat completions API."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from adarubric.core.exceptions import ConfigurationError
from adarubric.llm.openai_client import OpenAIClient


class OpenRouterClient(OpenAIClient):
    """Use OpenRouter model IDs and credentials, independently of OpenAI settings.

    Explicit arguments take precedence over OPENROUTER_API_KEY and
    OPENROUTER_BASE_URL. The default endpoint is https://openrouter.ai/api/v1.
    Structured generation requires a model that supports JSON object mode.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        reasoning_enabled: bool | None = None,
    ) -> None:
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ConfigurationError("Set OPENROUTER_API_KEY or pass llm.api_key for OpenRouter")
        super().__init__(
            model=model,
            api_key=key,
            base_url=base_url or os.environ.get("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1",
            max_retries=max_retries,
            extra_body=(
                {"reasoning": {"enabled": reasoning_enabled}}
                if reasoning_enabled is not None else None
            ),
        )
