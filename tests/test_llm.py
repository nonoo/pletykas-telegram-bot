import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import pytest

from llm import LLMClient, compress_image
from params import Params
from state import StateManager

@pytest.fixture(autouse=True)
def isolate_test_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

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


@pytest.mark.asyncio
async def test_generate_image_openai_route_generates_and_edits():
    p = Params()
    p.model_image_name = "openai/gpt-image-1"
    p.model_image_api_base = "https://api.test/v1"
    p.model_image_api_key = "img-key"
    client = LLMClient(p, StateManager("test.json"))

    # No source image -> fresh generation via /images/generations
    gen_body = json.dumps({"data": [{"b64_json": base64.b64encode(b"gen-bytes").decode()}]})
    session = _fake_session(lambda url, **kw: _embedding_resp(200, gen_body))
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        assert await client.generate_image("a cat") == b"gen-bytes"
    assert [call.args[0] for call in session.post.call_args_list] == ["https://api.test/v1/images/generations"]

    # Source image -> single edit request via /images/edits, no follow-up generation
    edit_resp = MagicMock(
        status=200,
        json=AsyncMock(return_value={"data": [{"b64_json": base64.b64encode(b"edited-bytes").decode()}]}),
    )
    session = _fake_session(lambda url, **kw: edit_resp)
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        assert await client.generate_image("a cat", base_image_bytes=b"src") == b"edited-bytes"
    assert [call.args[0] for call in session.post.call_args_list] == ["https://api.test/v1/images/edits"]


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
        text, reaction, img_spec, sched_spec = await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="trans",
            bot_username="pletykas_bot",
            is_direct_trigger=False,
        )
        assert text is None
        assert reaction is None
        assert img_spec is None
        assert sched_spec is None



def test_check_incapable_retry():
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)

    assert client._check_incapable_retry("<RETRY_WITH_LARGE_MODEL>")[0] is True
    assert client._check_incapable_retry("<RETRY_WITH_LARGE_MODEL: needs vision>")[1] == "needs vision"
    assert client._check_incapable_retry("<NEED_LARGE_MODEL>")[0] is True
    assert client._check_incapable_retry("Sajnos nem tudok rákeresni az interneten a mai meccsre.")[0] is False
    assert client._check_incapable_retry("I cannot search the web for real-time information.")[0] is False
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
            return "<RETRY_WITH_LARGE_MODEL: needs vision>", 10, 5
        else:
            assert model_name == "smart-large-model"
            assert api_key == "large-key"
            return "Here is the large model answer!", 50, 20

    with patch.object(client, "_call_openai_compatible", AsyncMock(side_effect=mock_call_openai)):
        text, reaction, img_spec, sched_spec = await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="Who won today?",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
        )
        assert call_count == 2
        assert text == "Here is the large model answer!"

def test_extract_web_tools():
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)

    cleaned, queries, urls = client._extract_web_tools(
        "Nézzük meg! <WEB_SEARCH:weather Budapest today> <fetch_url:https://example.com/page>"
    )
    assert cleaned == "Nézzük meg!"
    assert queries == ["weather Budapest today"]
    assert urls == ["https://example.com/page"]

    cleaned2, queries2, urls2 = client._extract_web_tools("<WEB_SEARCH:   > <WEB_SEARCH:a> <WEB_SEARCH:b>")
    assert cleaned2 == ""
    assert queries2 == ["a", "b"]
    assert urls2 == []

    assert client._extract_web_tools("plain reply") == ("plain reply", [], [])
    assert client._extract_web_tools("") == ("", [], [])


@pytest.mark.asyncio
async def test_evaluate_and_reply_runs_one_web_tool_round():
    p = Params()
    p.model_name = "small-model"
    p.model_api_base = "https://small.api.com"
    p.model_api_key = "small-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    prompts = []

    async def mock_call_openai(api_base, api_key, model_name, messages, **kwargs):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            return "Fogalmam sincs. <WEB_SEARCH:weather Budapest today>", 10, 5
        return "Budapesten 17 fok van! ☀️", 20, 10

    tool_block = 'Search results for "weather Budapest today":\n1. Időkép\n   https://www.idokep.hu/\n   17°C'
    with patch.object(client, "_call_openai_compatible", AsyncMock(side_effect=mock_call_openai)), \
         patch.object(client, "_run_web_tools", AsyncMock(return_value=tool_block)) as mock_tools:
        text, reaction, img_spec, sched_spec = await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="milyen idő van Budapesten?",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
        )

    mock_tools.assert_awaited_once_with(["weather Budapest today"], [])
    assert len(prompts) == 2
    assert "[Web Tool Results]" in prompts[1]
    assert "https://www.idokep.hu/" in prompts[1]
    assert text == "Budapesten 17 fok van! ☀️"


@pytest.mark.asyncio
async def test_evaluate_and_reply_without_tool_tags_skips_tool_round():
    p = Params()
    p.model_name = "small-model"
    p.model_api_base = "https://small.api.com"
    p.model_api_key = "small-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    async def mock_call_openai(api_base, api_key, model_name, messages, **kwargs):
        return "Szia! Milyen szép nap van ma!", 10, 5

    with patch.object(client, "_call_openai_compatible", AsyncMock(side_effect=mock_call_openai)), \
         patch.object(client, "_run_web_tools", AsyncMock()) as mock_tools:
        text, _, _, _ = await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="szia",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
        )

    mock_tools.assert_not_awaited()
    assert text == "Szia! Milyen szép nap van ma!"


@pytest.mark.asyncio
async def test_evaluate_and_reply_strips_tool_tags_from_followup():
    p = Params()
    p.model_name = "small-model"
    p.model_api_base = "https://small.api.com"
    p.model_api_key = "small-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    prompts = []

    async def mock_call_openai(api_base, api_key, model_name, messages, **kwargs):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            return "<FETCH_URL:https://example.com/page>", 10, 5
        return "Itt a tartalom! <WEB_SEARCH:more> <FETCH_URL:https://example.com/other>", 20, 10

    with patch.object(client, "_call_openai_compatible", AsyncMock(side_effect=mock_call_openai)), \
         patch.object(client, "_run_web_tools", AsyncMock(return_value="Content of https://example.com/page:\nHello")):
        text, _, _, _ = await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="olvasd el",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
        )

    # Exactly one tool round: the follow-up's own tool tags are stripped, never executed
    assert len(prompts) == 2
    assert text == "Itt a tartalom!"
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
        text, reaction, img_spec, sched_spec = await client.evaluate_and_reply(
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
        text, reaction, img_spec, desc, sched_spec = await client.describe_and_reply_image(
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
        text, reaction, img_spec, desc, sched_spec = await client.describe_and_reply_image(
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
        text, reaction, img_spec, desc, sched_spec = await client.describe_and_reply_image(
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
        text, reaction, img_spec, desc, sched_spec = await client.describe_and_reply_image(
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
async def test_curate_memory_archive_promote_keys_and_prompt_sections():
    p = Params()
    p.model_name = "gemini-2.5-flash"
    p.model_api_base = ""
    p.model_api_key = "test-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    mock_llm_output = json.dumps({
        "facts_to_archive": ["Old topic"],
        "facts_to_promote": ["Archived topic"],
        "dynamics_to_archive": ["Alice & Bob"],
        "dynamics_to_promote": ["Carl & Dana"],
        "jokes_to_archive": ["Old joke"],
        "jokes_to_promote": ["Old archived joke"],
    })

    deep = {"memories": [{"topic": "Archived topic", "content": "x"}], "dynamics": [], "inside_jokes": []}
    candidates = {"facts": ["Old topic"], "dynamics": [], "jokes": []}

    with patch.object(client, "_call_genai", AsyncMock(return_value=(mock_llm_output, 10, 10))) as mock_call:
        res = await client.curate_memory(
            {"memories": [], "dynamics": [], "inside_jokes": []},
            "transcript",
            deep_memories=deep,
            archive_candidates=candidates,
            archive_age_days=3,
        )

    assert res["facts_to_archive"] == ["Old topic"]
    assert res["facts_to_promote"] == ["Archived topic"]
    assert res["dynamics_to_archive"] == ["Alice & Bob"]
    assert res["dynamics_to_promote"] == ["Carl & Dana"]
    assert res["jokes_to_archive"] == ["Old joke"]
    assert res["jokes_to_promote"] == ["Old archived joke"]

    prompt = mock_call.call_args[1]["contents"][0]
    assert "[Deep Memory Entries (archived, read-only reference)]" in prompt
    assert "Archived topic" in prompt
    assert "[Archive Candidates (hot entries older than 3 days)]" in prompt
    assert "Entries older than 3 days SHOULD be moved to the archive" in prompt


@pytest.mark.asyncio
async def test_curate_memory_backward_compat_missing_keys():
    p = Params()
    p.model_name = "gemini-2.5-flash"
    p.model_api_base = ""
    p.model_api_key = "test-key"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    # Old-model output without archive/promote keys => empty lists, no crash
    mock_output = '{"facts_to_add": [{"topic": "Alice", "content": "Lives in Berlin"}]}'
    with patch.object(client, "_call_genai", AsyncMock(return_value=(mock_output, 10, 10))) as mock_call:
        res = await client.curate_memory({"memories": [], "dynamics": [], "inside_jokes": []}, "transcript")

    for key in ("facts_to_archive", "facts_to_promote", "dynamics_to_archive",
                "dynamics_to_promote", "jokes_to_archive", "jokes_to_promote"):
        assert res[key] == []
    assert len(res["facts_to_add"]) == 1

    # No deep store passed and no candidates => sections omitted, but the default
    # age rule (7 days) is still rendered
    prompt = mock_call.call_args[1]["contents"][0]
    assert "[Deep Memory Entries" not in prompt
    assert "[Archive Candidates" not in prompt
    assert "Entries older than 7 days SHOULD be moved to the archive" in prompt

    # 0 days disables the age rule entirely
    with patch.object(client, "_call_genai", AsyncMock(return_value=(mock_output, 10, 10))) as mock_call0:
        await client.curate_memory({"memories": [], "dynamics": [], "inside_jokes": []}, "transcript", archive_age_days=0)
    prompt0 = mock_call0.call_args[1]["contents"][0]
    assert "SHOULD be moved to the archive" not in prompt0


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

def test_extract_schedule():
    p = Params()
    s = StateManager("test.json")
    client = LLMClient(p, s)

    # 1. Oneshot schedule
    text1 = """Rendben, észben tartom!
<SCHEDULE:oneshot>
Time: in 2 hours
Description: Remind Alice about test report
</SCHEDULE:oneshot>"""
    cleaned1, spec1 = client._extract_schedule(text1)
    assert cleaned1 == "Rendben, észben tartom!"
    assert spec1 is not None
    assert spec1["action"] == "create"
    assert spec1["type"] == "oneshot"
    assert spec1["time"] == "in 2 hours"
    assert spec1["description"] == "Remind Alice about test report"

    # 2. Periodic schedule with start
    text2 = """<SCHEDULE:periodic>
Interval: 1 month
Start: 2026-10-01 10:00:00
Description: Check monthly expenses
</SCHEDULE:periodic>
Beállítottam a havi emlékeztetőt!"""
    cleaned2, spec2 = client._extract_schedule(text2)
    assert cleaned2 == "Beállítottam a havi emlékeztetőt!"
    assert spec2 is not None
    assert spec2["action"] == "create"
    assert spec2["type"] == "periodic"
    assert spec2["interval"] == "1 month"
    assert spec2["start"] == "2026-10-01 10:00:00"
    assert spec2["description"] == "Check monthly expenses"

    # 3. Cancel with ID in tag
    text3 = "Rendben, töröltem! <SCHEDULE:cancel:sched_1727339000_1042>"
    cleaned3, spec3 = client._extract_schedule(text3)
    assert cleaned3 == "Rendben, töröltem!"
    assert spec3 is not None
    assert spec3["action"] == "cancel"
    assert spec3["schedule_id"] == "sched_1727339000_1042"

    # 4. Cancel with body
    text4 = "Törölve! <SCHEDULE:cancel>sched_999</SCHEDULE:cancel>"
    cleaned4, spec4 = client._extract_schedule(text4)
    assert cleaned4 == "Törölve!"
    assert spec4 is not None
    assert spec4["action"] == "cancel"
    assert spec4["schedule_id"] == "sched_999"

    # 5. No schedule
    text5 = "Csak egy sima üzenet."
    cleaned5, spec5 = client._extract_schedule(text5)
    assert cleaned5 == "Csak egy sima üzenet."
    assert spec5 is None


@pytest.mark.asyncio
async def test_evaluate_and_reply_with_schedule_tag():
    p = Params()
    p.model_name = "gemini-2.5-flash"
    p.model_api_base = ""
    s = StateManager("test.json")
    client = LLMClient(p, s)

    model_output = """Persze, szólok majd!
<SCHEDULE:oneshot>
Time: +30m
Description: Szólj Bélának hogy indul a busz
</SCHEDULE:oneshot>"""

    with patch.object(client, "_call_genai", AsyncMock(return_value=(model_output, 50, 20))):
        text, reaction, img_spec, sched_spec = await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="trans",
            bot_username="pletykas_bot",
            is_direct_trigger=True,
        )
        assert text == "Persze, szólok majd!"
        assert reaction is None
        assert img_spec is None
        assert sched_spec is not None
        assert sched_spec["action"] == "create"
        assert sched_spec["time"] == "+30m"
        assert sched_spec["description"] == "Szólj Bélának hogy indul a busz"


@pytest.mark.asyncio
async def test_generate_scheduled_reply():
    p = Params()
    p.model_name = "gemini-2.5-flash"
    p.model_api_base = ""
    s = StateManager("test.json")
    client = LLMClient(p, s)

    captured_prompt = None
    async def mock_call_genai(api_key, model_name, contents, **kwargs):
        nonlocal captured_prompt
        captured_prompt = contents[0]
        return "Hahó Béla! Indul a buszod 5 perc múlva! 🚌", 40, 15

    with patch.object(client, "_call_genai", AsyncMock(side_effect=mock_call_genai)):
        reply = await client.generate_scheduled_reply(
            system_prompt="System prompt test",
            memory_context="Memory test",
            transcript="Transcript test",
            scheduled_description="Szólj Bélának hogy indul a busz",
            scheduled_type="oneshot",
            timezone_str="Europe/Budapest",
        )
        assert reply == "Hahó Béla! Indul a buszod 5 perc múlva! 🚌"
        assert captured_prompt is not None
        assert "[Scheduled Task Execution]" in captured_prompt
        assert "Type: oneshot" in captured_prompt
        assert "Szólj Bélának hogy indul a busz" in captured_prompt


def test_extract_poll_hungarian_with_intro_text():
    from llm import extract_poll
    text = """Srácok, Norbi este repülőtér + hotel Akossal, holnap meg Birmingham — szerintetek ki fogja előbb feladni a sorozást: ő vagy Misa? 😏

<POLL>
Question: Ki bírja tovább a birminghami sorozást?
Options:
- Norbi
- Misa
- GGabor
- Santha Gergő
</POLL>"""
    cleaned, spec = extract_poll(text)
    assert cleaned == "Srácok, Norbi este repülőtér + hotel Akossal, holnap meg Birmingham — szerintetek ki fogja előbb feladni a sorozást: ő vagy Misa? 😏"
    assert spec is not None
    assert spec["question"] == "Ki bírja tovább a birminghami sorozást?"
    assert spec["options"] == ["Norbi", "Misa", "GGabor", "Santha Gergő"]


def test_extract_poll_english_standard():
    from llm import extract_poll
    text = """<POLL>
Question: What is your favorite programming language?
Options:
- Python
- Rust
- Go
</POLL>"""
    cleaned, spec = extract_poll(text)
    assert cleaned == ""
    assert spec is not None
    assert spec["question"] == "What is your favorite programming language?"
    assert spec["options"] == ["Python", "Rust", "Go"]


def test_extract_poll_without_prefix_and_numbered_options():
    from llm import extract_poll
    text = """Hova menjünk ebédelni?

<POLL>
Melyik étterem legyen mára?
1. Burger King
2. Wasabi
3. Pizza Forte
</POLL>

Szavazzatok délután 1-ig!"""
    cleaned, spec = extract_poll(text)
    assert "Hova menjünk ebédelni?" in cleaned
    assert "Szavazzatok délután 1-ig!" in cleaned
    assert "<POLL>" not in cleaned
    assert "</POLL>" not in cleaned
    assert spec is not None
    assert spec["question"] == "Melyik étterem legyen mára?"
    assert spec["options"] == ["Burger King", "Wasabi", "Pizza Forte"]


def test_extract_poll_malformed_and_unclosed():
    from llm import extract_poll
    # Single option -> invalid poll
    text = "Hello! <POLL>\nQuestion: Single option?\n- Only me\n</POLL>"
    cleaned, spec = extract_poll(text)
    assert cleaned == "Hello!"
    assert spec is None

    # Unclosed poll tag
    text2 = "Hello! <POLL>\nQuestion: Valid question?\n- Opt 1\n- Opt 2"
    cleaned2, spec2 = extract_poll(text2)
    assert cleaned2 == "Hello!"
    assert spec2 is not None
    assert spec2["question"] == "Valid question?"
    assert spec2["options"] == ["Opt 1", "Opt 2"]


def test_extract_poll_deduplication_and_limits():
    from llm import extract_poll
    text = """<POLL>
Question: """ + ("Q" * 400) + """
Options:
- 'Apple'
- "Banana"
- Apple
""" + "\n".join([f"- Option {i}" for i in range(15)]) + """
</POLL>"""
    cleaned, spec = extract_poll(text)
    assert spec is not None
    assert len(spec["question"]) == 300
    # Deduplicated 'Apple', max 10 options
    assert len(spec["options"]) == 10
    assert spec["options"][0] == "Apple"
    assert spec["options"][1] == "Banana"
    assert spec["options"][2] == "Option 0"


def test_extract_forget_structured():
    from llm import extract_forget
    text = """Már el is felejtettem! 😉
<FORGET>
Facts to Discard:
- Alice: Lives in Berlin
- Bob
Facts to Update:
- Topic: Charlie
  Content: Senior frontend developer
Dynamics to Discard:
- Alice & Bob
Jokes to Discard:
- Tab War
</FORGET>
Semmi ilyenre nem emlékszem!"""

    cleaned, spec = extract_forget(text)
    assert "<FORGET>" not in cleaned
    assert "</FORGET>" not in cleaned
    assert "Már el is felejtettem! 😉" in cleaned
    assert "Semmi ilyenre nem emlékszem!" in cleaned
    assert spec is not None
    assert spec["clear_all"] is False
    assert spec["facts_to_discard"] == ["Alice: Lives in Berlin", "Bob"]
    assert spec["facts_to_update"] == [{"topic": "Charlie", "content": "Senior frontend developer"}]
    assert spec["dynamics_to_discard"] == ["Alice & Bob"]
    assert spec["jokes_to_discard"] == ["Tab War"]


def test_extract_forget_all():
    from llm import extract_forget
    text = "Teljes amnézia!\n<FORGET>\nALL\n</FORGET>"
    cleaned, spec = extract_forget(text)
    assert cleaned == "Teljes amnézia!"
    assert spec["clear_all"] is True

    text2 = "<FORGET:all>"
    cleaned2, spec2 = extract_forget(text2)
    assert cleaned2 == ""
    assert spec2["clear_all"] is True


def test_extract_forget_inline_and_bullets():
    from llm import extract_forget
    text = "Rendben! <FORGET:Berlin>\nViszlát!"
    cleaned, spec = extract_forget(text)
    assert "<FORGET" not in cleaned
    assert "Rendben!" in cleaned
    assert "Viszlát!" in cleaned
    assert "Berlin" in spec["raw_targets"]

    text2 = """<FORGET>
- Alice moved to Berlin
- Pineapple pizza
</FORGET>"""
    cleaned2, spec2 = extract_forget(text2)
    assert spec2["raw_targets"] == ["Alice moved to Berlin", "Pineapple pizza"]


@pytest.mark.asyncio
async def test_curate_forget_genai():
    p = Params()
    p.model_name = "gemini-2.5-flash"
    p.model_api_base = ""
    s = StateManager("test.json")
    client = LLMClient(p, s)

    mock_output = """```json
{
  "clear_all": false,
  "facts_to_discard": ["Alice"],
  "facts_to_update": [{"topic": "Bob", "content": "Lives in Munich"}],
  "dynamics_to_discard": ["Alice & Bob"],
  "jokes_to_discard": ["Berlin Wall"]
}
```"""
    with patch.object(client, "_call_genai", AsyncMock(return_value=(mock_output, 100, 40))):
        res = await client.curate_forget(
            current_memories={"memories": [], "dynamics": [], "inside_jokes": []},
            forget_request_text="Felejtsd el Berlint és Alice-t!",
            sender_name="Bob",
            recent_transcript="Bob: felejtsd el Alice-t",
        )
        assert res["clear_all"] is False
        assert res["facts_to_discard"] == ["Alice"]
        assert res["facts_to_update"] == [{"topic": "Bob", "content": "Lives in Munich"}]
        assert res["dynamics_to_discard"] == ["Alice & Bob"]
        assert res["jokes_to_discard"] == ["Berlin Wall"]


@pytest.mark.asyncio
async def test_curate_forget_openai():
    p = Params()
    p.model_name = "custom-llm"
    p.model_api_base = "https://custom.api.com"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    mock_output = json.dumps({
        "clear_all": True,
        "facts_to_discard": [],
        "facts_to_update": [],
        "dynamics_to_discard": [],
        "jokes_to_discard": [],
    })
    with patch.object(client, "_call_openai_compatible", AsyncMock(return_value=(mock_output, 80, 25))):
        res = await client.curate_forget(
            current_memories={"memories": [], "dynamics": [], "inside_jokes": []},
            forget_request_text="Töröld az összes emléked!",
            sender_name="Admin",
        )
        assert res["clear_all"] is True


@pytest.mark.asyncio
async def test_curate_forget_failure_returns_none():
    p = Params()
    p.model_name = "custom-llm"
    p.model_api_base = "https://custom.api.com"
    s = StateManager("test.json")
    client = LLMClient(p, s)

    # Unparseable response -> None, so the caller can keep an inline spec
    with patch.object(client, "_call_openai_compatible", AsyncMock(return_value=("no json here", 10, 5))):
        res = await client.curate_forget(
            current_memories={"memories": [], "dynamics": [], "inside_jokes": []},
            forget_request_text="Felejtsd el Berlint!",
            sender_name="Bob",
        )
        assert res is None

    # Model call failure -> None
    with patch.object(client, "_call_openai_compatible", AsyncMock(side_effect=RuntimeError("boom"))):
        res = await client.curate_forget(
            current_memories={"memories": [], "dynamics": [], "inside_jokes": []},
            forget_request_text="Felejtsd el Berlint!",
            sender_name="Bob",
        )
        assert res is None


def _embedding_resp(status, body):
    return MagicMock(status=status, text=AsyncMock(return_value=body))


def _fake_session(post_impl):
    """Session whose post() delegates to post_impl(url, **kwargs) -> response object."""
    session = MagicMock()

    def post(url, **kwargs):
        resp = post_impl(url, **kwargs)
        return AsyncMock(__aenter__=AsyncMock(return_value=resp), __aexit__=AsyncMock())

    session.post = MagicMock(side_effect=post)
    return session


def _embed_client(dim=0, api_base="https://api.test"):
    p = Params()
    p.model_api_base = api_base
    p.model_api_key = "k"
    p.model_embed_dim = dim
    s = StateManager("test.json")
    return LLMClient(p, s)


@pytest.mark.asyncio
async def test_embed_texts_happy_path_sorted_and_dimensions():
    body = json.dumps({"data": [
        {"index": 1, "embedding": [0.0, 1.0]},
        {"index": 0, "embedding": [1.0, 0.0]},
    ]})
    client = _embed_client(dim=0)
    session = _fake_session(lambda url, **kw: _embedding_resp(200, body))
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        vectors = await client.embed_texts(["first", "second"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]  # sorted by index even when data arrives out of order
    assert session.post.call_args.args[0] == "https://api.test/embeddings"
    payload = session.post.call_args.kwargs["json"]
    assert payload["model"] == "google/gemini-embedding-2"
    assert payload["input"] == ["first", "second"]
    assert "dimensions" not in payload

    client_dim = _embed_client(dim=768)
    session_dim = _fake_session(lambda url, **kw: _embedding_resp(200, body))
    with patch.object(client_dim, "_get_session", AsyncMock(return_value=session_dim)):
        await client_dim.embed_texts(["first", "second"])
    assert session_dim.post.call_args.kwargs["json"]["dimensions"] == 768


@pytest.mark.asyncio
async def test_embed_texts_degraded_paths_return_empty():
    client = _embed_client()

    # Count mismatch (1 vector for 2 texts) -> []
    session = _fake_session(lambda url, **kw: _embedding_resp(200, json.dumps({"data": [{"index": 0, "embedding": [1.0]}]})))
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        assert await client.embed_texts(["a", "b"]) == []

    # HTTP 500 -> []
    session = _fake_session(lambda url, **kw: _embedding_resp(500, "server error"))
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        assert await client.embed_texts(["a"]) == []

    # Empty input -> [] without any HTTP call
    session = _fake_session(lambda url, **kw: _embedding_resp(200, "{}"))
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        assert await client.embed_texts([]) == []
    session.post.assert_not_called()

    # No embed api_base anywhere -> [] (fail fast, no malformed URL request)
    client_no_base = _embed_client(api_base="")
    session = _fake_session(lambda url, **kw: _embedding_resp(200, "{}"))
    with patch.object(client_no_base, "_get_session", AsyncMock(return_value=session)):
        assert await client_no_base.embed_texts(["a"]) == []
    session.post.assert_not_called()


@pytest.mark.asyncio
async def test_embed_texts_timeout_uses_5s_budget():
    client = _embed_client()

    def post_impl(url, **kw):
        raise asyncio.TimeoutError()

    session = _fake_session(post_impl)
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        assert await client.embed_texts(["x"]) == []
    assert session.post.call_args.kwargs["timeout"].total == 5


@pytest.mark.asyncio
async def test_embed_texts_chunks_batches_over_64():
    client = _embed_client()
    texts = [f"t{i}" for i in range(130)]

    calls = []

    def post_impl(url, **kw):
        payload = kw["json"]
        calls.append(payload["input"])
        body = json.dumps({"data": [
            {"index": i, "embedding": [float(len(t))]} for i, t in enumerate(payload["input"])
        ]})
        return _embedding_resp(200, body)

    session = _fake_session(post_impl)
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        vectors = await client.embed_texts(texts)

    # Split into provider-safe chunks, preserving input order across chunks
    assert [len(c) for c in calls] == [64, 64, 2]
    assert calls[0] + calls[1] + calls[2] == texts
    assert vectors == [[float(len(t))] for t in texts]

    # A failing chunk aborts the whole embed (no further requests)
    fail_calls = []

    def post_fail_second(url, **kw):
        fail_calls.append(kw["json"]["input"])
        if len(fail_calls) == 2:
            return _embedding_resp(500, "boom")
        n = len(kw["json"]["input"])
        body = json.dumps({"data": [{"index": i, "embedding": [1.0]} for i in range(n)]})
        return _embedding_resp(200, body)

    session2 = _fake_session(post_fail_second)
    with patch.object(client, "_get_session", AsyncMock(return_value=session2)):
        assert await client.embed_texts(texts) == []
    assert len(fail_calls) == 2


@pytest.mark.asyncio
async def test_chat_timeout_applied_and_curation_exempt():
    client = _embed_client()
    chat_body = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    session = _fake_session(lambda url, **kw: _embedding_resp(200, chat_body))
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        await client.evaluate_and_reply(
            system_prompt="sys",
            memory_context="mem",
            transcript="tr",
            bot_username="bot",
            is_direct_trigger=True,
            talkativeness=5,
        )
        assert session.post.call_args.kwargs["timeout"].total == 10

        await client.curate_memory({"memories": [], "dynamics": [], "inside_jokes": []}, "transcript")
        assert "timeout" not in session.post.call_args.kwargs


@pytest.mark.asyncio
async def test_chat_timeout_error_propagates_from_transport():
    client = _embed_client()

    def post_impl(url, **kw):
        raise asyncio.TimeoutError()

    session = _fake_session(post_impl)
    with patch.object(client, "_get_session", AsyncMock(return_value=session)):
        with pytest.raises(asyncio.TimeoutError):
            await client._call_openai_compatible(
                "https://api.test", "k", "model", [{"role": "user", "content": "hi"}],
                timeout_sec=10,
            )
