"""OpenRouter configuration and request regression tests (no live API calls)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from adarubric.config import AdaRubricConfig, LLMConfig
from adarubric.core.exceptions import ConfigurationError
from adarubric.llm import OpenRouterClient
from adarubric.pipeline import AdaRubricPipeline


class Answer(BaseModel):
    answer: str


@pytest.fixture
def sdk():
    with patch("adarubric.llm.openai_client.AsyncOpenAI") as factory:
        client = factory.return_value
        client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'))]
            )
        )
        client.close = AsyncMock()
        yield factory


async def test_pipeline_requests_and_close(sdk, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    config = AdaRubricConfig(llm=LLMConfig(provider="openrouter"))
    AdaRubricPipeline.from_config(config)
    sdk.assert_called_once_with(api_key="router-test", base_url="https://openrouter.ai/api/v1")
    async with OpenRouterClient(model="provider/model") as client:
        messages = [{"role": "user", "content": "Return an answer"}]
        assert await client.generate_text(messages) == '{"answer":"ok"}'
        assert await client.generate_structured(messages, Answer) == Answer(answer="ok")
        request = sdk.return_value.chat.completions.create.call_args.kwargs
        assert request["model"] == "provider/model"
        assert request["response_format"] == {"type": "json_object"}
        assert messages[0]["content"] == "Return an answer"
    sdk.return_value.close.assert_awaited_once()


def test_explicit_settings_override_environment(sdk, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://env.example/v1")
    OpenRouterClient(api_key="explicit", base_url="https://custom.example/v1")
    sdk.assert_called_once_with(api_key="explicit", base_url="https://custom.example/v1")


def test_environment_endpoint(sdk, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://env.example/v1")
    OpenRouterClient()
    sdk.assert_called_once_with(api_key="env-key", base_url="https://env.example/v1")


def test_missing_key_does_not_fall_back_to_openai(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider-key")
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        OpenRouterClient()


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_config_file_credentials(extension, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    path = tmp_path / f"config.{extension}"
    path.write_text(json.dumps({"llm": {"provider": "openrouter"}}))
    loader = getattr(AdaRubricConfig, f"from_{extension}")
    config = loader(path)
    assert config.llm.api_key == "router-key"
    assert config.llm.model == "openai/gpt-4o"
    path.write_text(json.dumps({"llm": {
        "provider": "openrouter", "api_key": "explicit", "model": "provider/model",
    }}))
    config = loader(path)
    assert config.llm.api_key == "explicit"
    assert config.llm.model == "provider/model"


def test_dotenv_credentials_and_endpoint(sdk, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=file-key\nOPENROUTER_BASE_URL=https://file.example/v1\n"
    )
    OpenRouterClient()
    sdk.assert_called_once_with(api_key="file-key", base_url="https://file.example/v1")
    # Ensure dotenv's process environment changes are undone by monkeypatch.
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.delenv("OPENROUTER_BASE_URL")


def test_environment_overrides_dotenv(sdk, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "shell-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://shell.example/v1")
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=file-key\nOPENROUTER_BASE_URL=https://file.example/v1\n"
    )
    OpenRouterClient()
    sdk.assert_called_once_with(api_key="shell-key", base_url="https://shell.example/v1")


def test_config_loads_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=file-key\n")
    path = tmp_path / "config.json"
    path.write_text('{"llm": {"provider": "openrouter"}}')
    assert AdaRubricConfig.from_json(path).llm.api_key == "file-key"
    monkeypatch.delenv("OPENROUTER_API_KEY")


@pytest.mark.parametrize("enabled", [False, True, None])
async def test_reasoning_setting_reaches_requests(sdk, enabled):
    config = AdaRubricConfig(llm=LLMConfig(
        provider="openrouter", model="deepseek/deepseek-chat-v3.1",
        api_key="test", reasoning_enabled=enabled,
    ))
    with patch("adarubric.pipeline.OpenRouterClient", wraps=OpenRouterClient) as factory:
        AdaRubricPipeline.from_config(config)
        assert factory.call_args.kwargs["reasoning_enabled"] is enabled
    async with OpenRouterClient(
        model=config.llm.model, api_key="test", reasoning_enabled=enabled,
    ) as client:
        messages = [{"role": "user", "content": "Return an answer"}]
        await client.generate_text(messages)
        await client.generate_structured(messages, Answer)
        for request in sdk.return_value.chat.completions.create.call_args_list:
            assert request.kwargs["model"] == "deepseek/deepseek-chat-v3.1"
            if enabled is None:
                assert "extra_body" not in request.kwargs
            else:
                assert request.kwargs["extra_body"] == {"reasoning": {"enabled": enabled}}


async def test_direct_deepseek(sdk, monkeypatch):
    from adarubric.llm import DeepSeekClient

    monkeypatch.setenv("DEEPSEEK_API_KEY", "direct-test")
    cfg = LLMConfig(provider="deepseek", reasoning_enabled=False)
    assert cfg.model == "deepseek-chat"
    AdaRubricPipeline.from_config(AdaRubricConfig(llm=cfg))
    sdk.assert_called_once_with(api_key="direct-test", base_url="https://api.deepseek.com")
    async with DeepSeekClient(reasoning_enabled=False) as client:
        await client.generate_structured([{"role": "user", "content": "Answer JSON"}], Answer)
        request = sdk.return_value.chat.completions.create.call_args.kwargs
        assert request["model"] == "deepseek-chat"
        assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_direct_deepseek_missing_key(monkeypatch, tmp_path):
    from adarubric.llm import DeepSeekClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient()
