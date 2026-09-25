import io
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import pytest

from llm import LLMClient, compress_image
from params import Params
from state import StateManager


def test_compress_image():
    # Create large 2000x1500 test image in memory
    img = Image.new("RGB", (2000, 1500), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    compressed = compress_image(raw_bytes, max_dim=1280, quality=85)
    with Image.open(io.BytesIO(compressed)) as out_img:
        assert out_img.format == "JPEG"
        assert max(out_img.size) <= 1280

def test_llm_client_initialization_and_genai_caching():
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)
    assert hasattr(client, "_genai_clients")
    assert isinstance(client._genai_clients, dict)

    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        c1 = client._get_genai_client("test-api-key")
        assert c1 == mock_instance
        assert "test-api-key" in client._genai_clients
        # Second call returns cached instance
        c2 = client._get_genai_client("test-api-key")
        assert c2 == mock_instance
        mock_client_cls.assert_called_once_with(api_key="test-api-key")

def test_thinking_level_instruction_and_config():
    from google.genai import types
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)

    # Default (empty) -> instruct model not to think and set budget 0
    inst_empty = client._get_thinking_instruction("")
    assert "Do not think" in inst_empty
    cfg_empty = client._build_thinking_config("")
    assert cfg_empty.thinking_budget == 0

    # "off" / "none" / "0" / "disabled"
    assert "Do not think" in client._get_thinking_instruction("off")
    assert client._build_thinking_config("disabled").thinking_budget == 0

    # Numeric budget
    assert client._build_thinking_config("2048").thinking_budget == 2048

    # Named level
    inst_med = client._get_thinking_instruction("medium")
    assert "medium level" in inst_med
    cfg_med = client._build_thinking_config("medium")
    assert cfg_med.thinking_level == types.ThinkingLevel.MEDIUM

@pytest.mark.asyncio
async def test_openai_compatible_thinking_payload():
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)

    async def fake_post(url, headers=None, json=None):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value='{"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}')
        return mock_resp

    session = MagicMock()
    session.post = MagicMock(side_effect=lambda url, headers=None, json=None: AsyncMock(
        __aenter__=AsyncMock(return_value=MagicMock(
            status=200,
            text=AsyncMock(return_value='{"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}')
        )),
        __aexit__=AsyncMock()
    ))

    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        # 1. Empty thinking level -> reasoning_effort: "none"
        await client._call_openai_compatible("https://api.test", "key", "model", [{"role": "user", "content": "hi"}], thinking_level="")
        payload = session.post.call_args.kwargs["json"]
        assert payload.get("reasoning_effort") == "none"

        # 2. "off" -> reasoning_effort: "none"
        await client._call_openai_compatible("https://api.test", "key", "model", [{"role": "user", "content": "hi"}], thinking_level="off")
        payload = session.post.call_args.kwargs["json"]
        assert payload.get("reasoning_effort") == "none"

        # 3. "medium" -> reasoning_effort: "medium"
        await client._call_openai_compatible("https://api.test", "key", "model", [{"role": "user", "content": "hi"}], thinking_level="medium")
        payload = session.post.call_args.kwargs["json"]
        assert payload.get("reasoning_effort") == "medium"

        # 4. Digits "1024" -> thinking: {type: enabled, budget_tokens: 1024}
        await client._call_openai_compatible("https://api.test", "key", "model", [{"role": "user", "content": "hi"}], thinking_level="1024")
        payload = session.post.call_args.kwargs["json"]
        assert payload.get("thinking") == {"type": "enabled", "budget_tokens": 1024}


def test_extract_reaction():
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)

    # Simple emoji reaction
    text1, r1 = client._extract_reaction("That was amazing! <REACTION:🔥>")
    assert text1 == "That was amazing!"
    assert r1 == ("🔥", None)

    # Reaction with target message ID
    text2, r2 = client._extract_reaction("Hilarious! <REACTION:😂:1045> Good one!")
    assert text2 == "Hilarious!  Good one!"
    assert r2 == ("😂", 1045)

    # No reaction
    text3, r3 = client._extract_reaction("Just plain chat text.")
    assert text3 == "Just plain chat text."
    assert r3 is None


def test_extract_generate_image():
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)

    text = """Check this out!
<GENERATE_IMAGE>
Prompt: A neon cyberpunk kitten in Tokyo
Caption: Look what I found!
Source: 1042
Mode: modify
</GENERATE_IMAGE>
Hope you like it!"""

    cleaned, spec = client._extract_generate_image(text)
    assert "Check this out!" in cleaned
    assert "Hope you like it!" in cleaned
    assert "<GENERATE_IMAGE>" not in cleaned
    assert spec is not None
    assert spec["prompt"] == "A neon cyberpunk kitten in Tokyo"
    assert spec["caption"] == "Look what I found!"
    assert spec["source"] == "1042"
    assert spec["mode"] == "modify"


def test_extract_image_description():
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)

    text = """Nice photo!
<IMAGE_DESCRIPTION>
A golden retriever sitting on grass under a sunny blue sky.
</IMAGE_DESCRIPTION>
He looks so happy!"""

    cleaned, desc = client._extract_image_description(text)
    assert "Nice photo!" in cleaned
    assert "He looks so happy!" in cleaned
    assert "<IMAGE_DESCRIPTION>" not in cleaned
    assert desc == "A golden retriever sitting on grass under a sunny blue sky."


def test_is_genai_model():
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)

    # If api_base is empty, gemini/gemma/imagen are genai models
    assert client._is_genai_model("gemini-2.5-flash", "") is True
    assert client._is_genai_model("gemma-4-31b-it", "") is True
    assert client._is_genai_model("imagen-3.0-generate-002", "") is True

    # If api_base is provided, they route through the OpenAI-compatible endpoint
    assert client._is_genai_model("gemini-2.5-flash", "https://custom.endpoint.com") is False
    assert client._is_genai_model("deepseek-chat", "https://api.deepseek.com") is False


@pytest.mark.asyncio
async def test_evaluate_and_reply_no_reply_behavior():
    p = Params()
    p.model_name = "gemini-2.5-flash"
    p.model_api_base = ""
    p.model_api_key = "test-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    # Mock _call_genai returning <NO_REPLY>
    with patch.object(client, "_call_genai", AsyncMock(return_value=("<NO_REPLY>", 100, 5))):
        text, reaction, img_spec = await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="trans",
            bot_username="pletykas_bot",
            is_direct_trigger=False,
        )
        assert text is None
        assert reaction is None
        assert img_spec is None



def test_check_incapable_retry():
    p = Params()
    s = StateManager("test.json")
    s.set_search_grounding_active(True)
    client = LLMClient(p, s)

    assert client._check_incapable_retry("<RETRY_WITH_LARGE_MODEL>")[0] is True
    assert client._check_incapable_retry("<RETRY_WITH_LARGE_MODEL: Google Search needed>")[1] == "Google Search needed"
    assert client._check_incapable_retry("<NEED_LARGE_MODEL>")[0] is True
    assert client._check_incapable_retry("Sajnos nem tudok rákeresni az interneten a mai meccsre.")[0] is True
    assert client._check_incapable_retry("Nincs internet-hozzáférésem ehhez a kéréshez.")[0] is True
    assert client._check_incapable_retry("I cannot search the web for real-time information.")[0] is True
    assert client._check_incapable_retry("Szia, nagyon szép napunk van ma!")[0] is False
    assert client._check_incapable_retry("<NO_REPLY>")[0] is False


@pytest.mark.asyncio
async def test_evaluate_and_reply_retries_with_large_model():
    p = Params()
    p.model_name = "fast-small-model"
    p.model_api_base = "https://small.api.com"
    p.model_api_key = "small-key"
    p.model_large_name = "smart-large-model"
    p.model_large_api_base = "https://large.api.com"
    p.model_large_api_key = "large-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    call_count = 0

    async def mock_call_openai(api_base, api_key, model_name, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            assert model_name == "fast-small-model"
            assert api_key == "small-key"
            return "<RETRY_WITH_LARGE_MODEL: needs google search>", 10, 5
        else:
            assert model_name == "smart-large-model"
            assert api_key == "large-key"
            return "Here is the live search result from the web!", 50, 20

    with patch.object(client, "_call_openai_compatible", AsyncMock(side_effect=mock_call_openai)):
        text, reaction, img_spec = await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="Who won today?",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
        )
        assert call_count == 2
        assert text == "Here is the live search result from the web!"
@pytest.mark.asyncio
async def test_evaluate_and_reply_retries_with_large_model_and_chat_history_image():
    import base64
    p = Params()
    p.model_name = "fast-small-model"
    p.model_api_base = "https://small.api.com"
    p.model_api_key = "small-key"
    p.model_large_name = "smart-large-model"
    p.model_large_api_base = "https://large.api.com"
    p.model_large_api_key = "large-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    fake_b64 = base64.b64encode(b"saved-history-image").decode("utf-8")
    s.append_chat_message({
        "id": 50,
        "text": "[Photo]",
        "media_type": "photo",
        "media_b64": fake_b64,
    })

    # Step 1: Small model says it needs image analysis
    async def mock_call_openai(api_base, api_key, model_name, messages, **kwargs):
        return "<RETRY_WITH_LARGE_MODEL:image_analysis>", 10, 5

    # Step 2: Large model is called via _call_vision_model with the retrieved image
    with patch.object(client, "_call_openai_compatible", AsyncMock(side_effect=mock_call_openai)), patch.object(
        client, "_call_vision_model", AsyncMock(return_value=("Két fekete cica van a képen!", 50, 10))
    ) as mock_vision:
        text, reaction, img_spec = await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="szoval mi van a kepen?",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
        )
        mock_vision.assert_awaited_once()
        kwargs = mock_vision.call_args.kwargs
        assert kwargs["model_name"] == "smart-large-model"
        assert kwargs["image_bytes"] == b"saved-history-image"
        assert text == "Két fekete cica van a képen!"

@pytest.mark.asyncio
async def test_describe_and_reply_image_large_model_default():
    p = Params()
    p.model_name = "small-model"
    p.model_large_name = "large-model"
    s = StateManager("test.json")
    assert s.is_image_interpretation_large_model() is True
    client = LLMClient(p, s)

    with patch.object(client, "_call_vision_model", AsyncMock(return_value=("<IMAGE_DESCRIPTION>A cute puppy</IMAGE_DESCRIPTION>Look at this dog!", 50, 10))) as mock_vision:
        text, reaction, img_spec, desc = await client.describe_and_reply_image(
            system_prompt="sys",
            memory_context="mem",
            transcript="trans",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
            talkativeness=5,
            image_bytes=b"fake-image-bytes",
            caption="my dog",
        )
        mock_vision.assert_awaited_once()
        # Large model is called directly
        assert mock_vision.call_args.kwargs["model_name"] == "large-model"
        assert text == "Look at this dog!"
        assert desc == "A cute puppy"

@pytest.mark.asyncio
async def test_describe_and_reply_image_small_model_when_disabled():
    p = Params()
    p.model_name = "small-model"
    p.model_large_name = "large-model"
    s = StateManager("test.json")
    s.set_image_interpretation_large_model(False)
    client = LLMClient(p, s)

    calls = []
    async def mock_vision_call(model_name, **kwargs):
        calls.append(model_name)
        if model_name == "small-model":
            return "<IMAGE_DESCRIPTION>Two cats on a tree</IMAGE_DESCRIPTION>Those cats are chilling!", 30, 5
        return "<IMAGE_DESCRIPTION>Large model desc</IMAGE_DESCRIPTION>Large model text", 100, 20

    with patch.object(client, "_call_vision_model", AsyncMock(side_effect=mock_vision_call)):
        text, reaction, img_spec, desc = await client.describe_and_reply_image(
            system_prompt="sys",
            memory_context="mem",
            transcript="trans",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
            talkativeness=5,
            image_bytes=b"fake-image-bytes",
            caption="cats",
        )
        assert calls == ["small-model"]
        assert text == "Those cats are chilling!"
        assert desc == "Two cats on a tree"

@pytest.mark.asyncio
async def test_describe_and_reply_image_small_model_fallback_to_large():
    p = Params()
    p.model_name = "small-model"
    p.model_large_name = "large-model"
    s = StateManager("test.json")
    s.set_image_interpretation_large_model(False)
    client = LLMClient(p, s)

    calls = []
    async def mock_vision_call(model_name, **kwargs):
        calls.append(model_name)
        if model_name == "small-model":
            # Small model returns retry tag or failure
            return "<RETRY_WITH_LARGE_MODEL: needs high resolution vision>", 10, 5
        return "<IMAGE_DESCRIPTION>High res details</IMAGE_DESCRIPTION>Identified with large model!", 100, 20

    with patch.object(client, "_call_vision_model", AsyncMock(side_effect=mock_vision_call)):
        text, reaction, img_spec, desc = await client.describe_and_reply_image(
            system_prompt="sys",
            memory_context="mem",
            transcript="trans",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
            talkativeness=5,
            image_bytes=b"fake-image-bytes",
            caption="complex chart",
        )
        assert calls == ["small-model", "large-model"]
        assert text == "Identified with large model!"
        assert desc == "High res details"

@pytest.mark.asyncio
async def test_describe_and_reply_image_with_generate_image():
    p = Params()
    p.model_name = "small-model"
    p.model_large_name = "large-model"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    vision_resp = """<REACTION:😈>

<GENERATE_IMAGE>
Prompt: Close-up of smiling RoboCop with hair
Caption: Here is your smiling RoboCop!
Source: reply
Mode: modify
</GENERATE_IMAGE>

<IMAGE_DESCRIPTION>
Original RoboCop figure
</IMAGE_DESCRIPTION>"""

    with patch.object(client, "_call_vision_model", AsyncMock(return_value=(vision_resp, 100, 50))):
        text, reaction, img_spec, desc = await client.describe_and_reply_image(
            system_prompt="sys",
            memory_context="mem",
            transcript="trans",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
            talkativeness=5,
            image_bytes=b"fake-image-bytes",
            caption="make him smile",
        )
        assert reaction == ("😈", None)
        assert img_spec is not None
        assert img_spec["prompt"] == "Close-up of smiling RoboCop with hair"
        assert img_spec["caption"] == "Here is your smiling RoboCop!"
        assert img_spec["mode"] == "modify"
        assert img_spec["source"] == "reply"
        assert desc == "Original RoboCop figure"
        assert text is None  # Since all content was extracted into tags

@pytest.mark.asyncio
async def test_curate_memory_json_extraction():
    p = Params()
    p.model_name = "gemini-2.5-flash"
    p.model_api_base = ""
    p.model_api_key = "test-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    mock_llm_output = """Here are the memory updates based on the conversation:
```json
{
  "facts_to_add": [{"topic": "Alice", "content": "Lives in Berlin"}],
  "facts_to_update": [],
  "facts_to_discard": ["Old fact about Alice"],
  "dynamics_to_add": [{"members": ["Alice", "Bob"], "relation": "Good friends"}],
  "dynamics_to_discard": [],
  "jokes_to_add": [{"title": "TabWar", "context": "Funny war about tabs"}],
  "jokes_to_discard": []
}
```
Hope this helps!"""

    with patch.object(client, "_call_genai", AsyncMock(return_value=(mock_llm_output, 200, 50))):
        res = await client.curate_memory({"memories": [], "dynamics": [], "inside_jokes": []}, "transcript")
        assert len(res["facts_to_add"]) == 1
        assert res["facts_to_add"][0]["topic"] == "Alice"
        assert res["facts_to_discard"] == ["Old fact about Alice"]
        assert len(res["dynamics_to_add"]) == 1
        assert len(res["jokes_to_add"]) == 1


@pytest.mark.asyncio
async def test_curate_memory_uses_small_model_200k_tokens():
    p = Params()
    p.model_name = "fast-model"
    p.model_api_base = "https://fast.api.com"
    p.model_api_key = "fast-key"
    p.model_large_name = "large-model"
    p.model_large_api_base = "https://large.api.com"
    p.model_large_api_key = "large-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    mock_output = '{"facts_to_add": [{"topic": "Dogs", "content": "Bark"}]}'
    mock_call = AsyncMock(return_value=(mock_output, 100, 20))
    with patch.object(client, "_call_openai_compatible", mock_call):
        res = await client.curate_memory({"memories": [], "dynamics": [], "inside_jokes": []}, "transcript")
        assert len(res["facts_to_add"]) == 1
        mock_call.assert_awaited_once()
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["model_name"] == "fast-model"
        assert call_kwargs["api_base"] == "https://fast.api.com"
        assert call_kwargs["api_key"] == "fast-key"
        assert call_kwargs["max_tokens"] == 200_000


@pytest.mark.asyncio
async def test_debug_mode_stdout_logging(capsys):
    p = Params()
    p.model_name = "test-model"
    p.model_api_base = "https://test.api.com"
    p.model_api_key = "test-key"
    s = StateManager("test.json")
    s.set_debug_mode(True)
    client = LLMClient(p, s)

    client._log_debug_payload("TEST_REQ", '{"prompt": "hello"}')
    captured = capsys.readouterr()
    assert "--- [DEBUG TEST_REQ] ---" in captured.out
    assert '{"prompt": "hello"}' in captured.out

@pytest.mark.asyncio
async def test_debug_mode_openai_compatible_response_with_content(capsys):
    p = Params()
    s = StateManager("test.json")
    s.set_debug_mode(True)
    client = LLMClient(p, s)

    raw_json = '{"id": "chatcmpl-test", "choices": [{"message": {"content": "Hello, world!"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}'

    session = MagicMock()
    session.post = MagicMock(side_effect=lambda url, headers=None, json=None: AsyncMock(
        __aenter__=AsyncMock(return_value=MagicMock(
            status=200,
            text=AsyncMock(return_value=raw_json)
        )),
        __aexit__=AsyncMock()
    ))

    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        content, p_tok, c_tok = await client._call_openai_compatible("https://api.test", "key", "model", [{"role": "user", "content": "hi"}])
        assert content == "Hello, world!"
        captured = capsys.readouterr()
        assert "--- [DEBUG LLM RESPONSE (200): https://api.test/chat/completions] ---" in captured.out
        expected_section = f"{raw_json}\nHello, world!"
        assert expected_section in captured.out

@pytest.mark.asyncio
async def test_debug_mode_genai_response_with_content(capsys):
    p = Params()
    s = StateManager("test.json")
    s.set_debug_mode(True)
    client = LLMClient(p, s)

    mock_resp = MagicMock()
    mock_resp.text = "GenAI reply text!"
    mock_resp.model_dump_json.return_value = '{"candidates": [{"content": "raw"}]}'
    mock_resp.usage_metadata.prompt_token_count = 15
    mock_resp.usage_metadata.candidates_token_count = 8

    mock_genai_client = AsyncMock()
    mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    with patch.object(client, "_get_genai_client", return_value=mock_genai_client):
        content, p_tok, c_tok = await client._call_genai("api-key", "gemini-2.5-flash", [{"role": "user", "parts": ["hi"]}])
        assert content == "GenAI reply text!"
        captured = capsys.readouterr()
        assert "--- [DEBUG GENAI SDK RESPONSE (gemini-2.5-flash)] ---" in captured.out
        expected_section = '{"candidates": [{"content": "raw"}]}\nGenAI reply text!'
        assert expected_section in captured.out
@pytest.mark.asyncio
async def test_generate_image_passes_image_size():
    p = Params()
    p.model_image_name = "imagen-3.0-generate-002"
    p.model_image_api_base = ""
    p.model_image_api_key = "img-key"
    p.model_image_size = "1K"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    mock_genai_client = AsyncMock()
    mock_generated_image = MagicMock()
    mock_generated_image.image.image_bytes = b"fake-jpeg-bytes"
    mock_resp = MagicMock()
    mock_resp.generated_images = [mock_generated_image]

    mock_genai_client.aio.models.generate_images = AsyncMock(return_value=mock_resp)
    with patch.object(client, "_get_genai_client", return_value=mock_genai_client):
        img_bytes = await client.generate_image("A cute cat")
        assert img_bytes == b"fake-jpeg-bytes"
        mock_genai_client.aio.models.generate_images.assert_awaited_once()
        _, kwargs = mock_genai_client.aio.models.generate_images.call_args
        assert kwargs["config"].image_size == "1K"


@pytest.mark.asyncio
async def test_generate_image_gemini_interactions_sdk():
    import base64
    p = Params()
    p.model_image_name = "gemini-3.1-flash-lite-image"
    p.model_image_api_base = ""
    p.model_image_api_key = "img-key"
    p.model_image_size = "1K"
    s = StateManager("test.json")
    s.set_debug_mode(True)
    client = LLMClient(p, s)
    mock_part = MagicMock()
    mock_part.type = "image"
    mock_part.data = base64.b64encode(b"interaction-jpeg-bytes").decode("utf-8")
    mock_step = MagicMock()
    mock_step.type = "model_output"
    mock_step.content = [mock_part]
    mock_interaction = MagicMock()
    mock_interaction.steps = [mock_step]

    mock_genai_client = AsyncMock()
    mock_genai_client.aio.interactions.create = AsyncMock(return_value=mock_interaction)

    with patch.object(client, "_get_genai_client", return_value=mock_genai_client), patch.object(client, "_log_debug_payload") as mock_debug_log:
        img_bytes = await client.generate_image("A futuristic city", base_image_bytes=b"prior-image-bytes")
        assert img_bytes == b"interaction-jpeg-bytes"
        mock_genai_client.aio.interactions.create.assert_awaited_once()
        _, kwargs = mock_genai_client.aio.interactions.create.call_args
        assert kwargs["model"] == "models/gemini-3.1-flash-lite-image"
        assert kwargs["generation_config"]["image_config"]["image_size"] == "1K"
        assert kwargs["response_modalities"] == ["image", "text"]
        assert mock_debug_log.call_count >= 2
        req_call = mock_debug_log.call_args_list[0]
        assert "Google Interactions SDK: models/gemini-3.1-flash-lite-image" in req_call[0][0]
