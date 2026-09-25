import os
import pytest
from params import Params


def test_params_default_values():
    p = Params()
    assert p.state_file == "pletykas-state.json"
    assert p.memory_file == "pletykas-memory.json"
    assert p.model_name == ""
    assert p.model_large_name == ""
    assert p.model_image_name == ""
    assert p.model_image_size == "1K"
    assert p.model_thinking_level == ""
    assert p.effective_large_thinking_level == ""
    assert p.effective_image_thinking_level == ""


def test_params_cli_parsing():
    p = Params()
    p.parse([
        "--bot-token", "test-token-123",
        "--group-chat-id", "-100987654321",
        "--admin-user-ids", "111,222, 333",
        "--model-api-key", "primary-key",
        "--model-name", "deepseek-chat",
        "--model-api-base", "https://api.deepseek.com",
        "--model-image-size", "2K",
        "--model-thinking-level", "minimal",
        "--model-large-thinking-level", "low",
        "--model-image-thinking-level", "medium",
    ])
    assert p.bot_token == "test-token-123"
    assert p.group_chat_id == -100987654321
    assert p.admin_user_ids == [111, 222, 333]
    assert p.model_api_key == "primary-key"
    assert p.model_image_size == "2K"
    assert p.model_thinking_level == "minimal"
    assert p.effective_large_thinking_level == "low"
    assert p.effective_image_thinking_level == "medium"
    assert p.is_admin(999) is False
    assert p.is_group_authorized(-100987654321) is True
    assert p.is_group_authorized(-100000000000) is False


def test_params_env_parsing(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "env-token")
    monkeypatch.setenv("GROUP_CHAT_ID", "-100111222333")
    monkeypatch.setenv("ADMIN_USERIDS", "444,555")
    monkeypatch.setenv("MODEL_API_KEY", "env-key")
    monkeypatch.setenv("MODEL_IMAGE_SIZE", "512x512")
    monkeypatch.setenv("MODEL_THINKING_LEVEL", "high")
    p = Params()
    p.parse([])
    assert p.bot_token == "env-token"
    assert p.group_chat_id == -100111222333
    assert p.admin_user_ids == [444, 555]
    assert p.model_api_key == "env-key"
    assert p.model_image_size == "512x512"
    assert p.model_thinking_level == "high"
    assert p.effective_large_thinking_level == "high"
    assert p.effective_image_thinking_level == "high"

def test_params_llm_api_key_fallback(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "fallback-key")
    monkeypatch.setenv("BOT_TOKEN", "env-token")
    monkeypatch.setenv("GROUP_CHAT_ID", "-100111")
    monkeypatch.setenv("ADMIN_USERIDS", "123")

    p = Params()
    p.parse([])
    assert p.model_api_key == "fallback-key"


def test_params_model_fallback_properties():
    p = Params()
    p.parse([
        "--bot-token", "token",
        "--group-chat-id", "-100",
        "--admin-user-ids", "1",
        "--model-api-key", "main-key",
        "--model-api-base", "https://api.main.com",
        "--model-large-name", "gemini-2.5-flash",
        "--model-image-name", "imagen-3.0-generate-002",
    ])
    assert p.effective_large_api_key == "main-key"
    assert p.effective_large_api_base == ""
    assert p.effective_image_api_key == "main-key"
    assert p.effective_image_api_base == ""

    # When explicit keys given
    p.parse([
        "--bot-token", "token",
        "--group-chat-id", "-100",
        "--admin-user-ids", "1",
        "--model-api-key", "main-key",
        "--model-large-api-key", "large-key",
        "--model-large-api-base", "https://api.large.com",
        "--model-image-api-key", "img-key",
        "--model-image-api-base", "https://api.img.com",
    ])
    assert p.effective_large_api_key == "large-key"
    assert p.effective_large_api_base == "https://api.large.com"
    assert p.effective_image_api_key == "img-key"
    assert p.effective_image_api_base == "https://api.img.com"


def test_params_validation_errors():
    p = Params()
    # Missing token
    with pytest.raises(ValueError, match="bot_token must be non-empty"):
        p.parse(["--group-chat-id", "-100", "--admin-user-ids", "1", "--model-api-key", "k"])

    # Zero group chat ID
    with pytest.raises(ValueError, match="group_chat_id must be non-zero"):
        p.parse(["--bot-token", "t", "--group-chat-id", "0", "--admin-user-ids", "1", "--model-api-key", "k"])

    # Missing admin IDs
    with pytest.raises(ValueError, match="admin_user_ids must contain at least one valid user ID"):
        p.parse(["--bot-token", "t", "--group-chat-id", "-100", "--admin-user-ids", "", "--model-api-key", "k"])

    # Missing API key
    with pytest.raises(ValueError, match="model_api_key must be non-empty"):
        p.parse(["--bot-token", "t", "--group-chat-id", "-100", "--admin-user-ids", "1", "--model-api-key", ""])
