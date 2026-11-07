"""DeepSeek backend using its OpenAI-compatible chat completions API."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from adarubric.core.exceptions import ConfigurationError
from adarubric.llm.openai_client import OpenAIClient


class DeepSeekClient(OpenAIClient):
    """Use DeepSeek model IDs and credentials, independently of OpenAI settings.

    Explicit arguments take precedence over DEEPSEEK_API_KEY and
    DEEPSEEK_BASE_URL. The default endpoint is https://api.deepseek.com.
    Structured generation requires a model that supports JSON object mode.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        reasoning_enabled: bool | None = None,
    ) -> None:
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ConfigurationError("Set DEEPSEEK_API_KEY or pass llm.api_key for DeepSeek")
        super().__init__(
            model=model,
            api_key=key,
            base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com",
            max_retries=max_retries,
            extra_body=(
                {"thinking": {"type": "enabled" if reasoning_enabled else "disabled"}}
                if reasoning_enabled is not None else None
            ),
        )
