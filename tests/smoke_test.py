import io
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import pytest

from handlers import BotHandlers
from llm import LLMClient
from memory import MemoryManager
from params import Params
from state import StateManager
from telegram.constants import ChatAction


@pytest.mark.asyncio
async def test_end_to_end_smoke_flow():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        mf = os.path.join(td, "mem.json")

        p = Params()
        p.bot_token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        p.group_chat_id = -1001234567890
        p.admin_user_ids = [42]
        p.model_api_key = "fake-key"

        state = StateManager(sf, group_chat_id=p.group_chat_id)
        state.load()

        memory = MemoryManager(mf)
        memory.load()

        llm = LLMClient(p, state)
        handlers = BotHandlers(p, state, memory, llm)

        mock_bot = MagicMock()
        mock_bot.id = 9999
        mock_bot.username = "pletykas_bot"
        mock_bot.first_name = "Pletykas"
        mock_bot.set_message_reaction = AsyncMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=2001))
        mock_bot.send_photo = AsyncMock(return_value=MagicMock(message_id=2002))
        mock_bot.send_chat_action = AsyncMock()

        mock_context = MagicMock()
        mock_context.bot = mock_bot
        mock_context.job_queue = MagicMock()

        # Step 1: Simulate regular human message
        msg1 = MagicMock(
            message_id=1001,
            from_user=MagicMock(id=100, full_name="Alice"),
            text="Has anyone tried the new bakery on Main Street?",
            caption=None,
            photo=[],
            voice=None,
            audio=None,
            video=None,
            video_note=None,
            document=None,
            sticker=None,
            animation=None,
            reply_to_message=None,
            entities=[],
            date=None,
        )
        update1 = MagicMock(
            effective_chat=MagicMock(id=p.group_chat_id, type="supergroup"),
            effective_message=msg1,
            effective_user=msg1.from_user,
        )
        await handlers.on_message(update1, mock_context)
        assert len(state.get_chat_history()) == 1

        # Step 2: Direct trigger message mentioning bot with reaction and text reply
        msg2 = MagicMock(
            message_id=1002,
            from_user=MagicMock(id=101, full_name="Bob"),
            text="@pletykas_bot What do you think about that bakery?",
            caption=None,
            photo=[],
            voice=None,
            audio=None,
            video=None,
            video_note=None,
            document=None,
            sticker=None,
            animation=None,
            reply_to_message=None,
            entities=[],
            date=None,
        )
        update2 = MagicMock(
            effective_chat=MagicMock(id=p.group_chat_id, type="supergroup"),
            effective_message=msg2,
            effective_user=msg2.from_user,
        )

        mock_eval_response = ("Their croissants are absolute perfection! <REACTION:🔥:1001>", 150, 30)
        with patch.object(llm, "_call_openai_compatible", AsyncMock(return_value=mock_eval_response)):
            await handlers._execute_evaluation(mock_context, p.group_chat_id, is_direct_trigger=True, trigger_msg_id=1002)

        # Verify reaction was dispatched to message 1001
        mock_bot.set_message_reaction.assert_awaited()
        # Verify text reply was sent
        mock_bot.send_message.assert_awaited()

        # Step 3: Image generation trigger (<GENERATE_IMAGE>)
        mock_img_eval_response = (
            "<GENERATE_IMAGE>\nPrompt: Golden buttery croissants on a bakery display\nCaption: Fresh from the oven!\nSource: new\nMode: generate\n</GENERATE_IMAGE>",
            180,
            40,
        )
        dummy_img = Image.new("RGB", (100, 100), color="orange")
        buf = io.BytesIO()
        dummy_img.save(buf, format="JPEG")
        dummy_bytes = buf.getvalue()

        with patch.object(llm, "_call_openai_compatible", AsyncMock(return_value=mock_img_eval_response)):
            with patch.object(llm, "generate_image", AsyncMock(return_value=dummy_bytes)):
                await handlers._execute_evaluation(mock_context, p.group_chat_id, is_direct_trigger=True, trigger_msg_id=1003)

        # Verify upload_photo chat action was dispatched strictly during generation
        mock_bot.send_chat_action.assert_awaited_with(chat_id=p.group_chat_id, action=ChatAction.UPLOAD_PHOTO)
        # Verify send_photo was called
        mock_bot.send_photo.assert_awaited()

        # Step 4: Trigger curation
        mock_curate_response = """```json
{
  "facts_to_add": [{"topic": "Bakery", "content": "The group loves croissants from Main Street bakery."}],
  "facts_to_update": [],
  "facts_to_discard": [],
  "dynamics_to_add": [],
  "dynamics_to_discard": [],
  "jokes_to_add": [],
  "jokes_to_discard": []
}
```"""
        with patch.object(llm, "_call_openai_compatible", AsyncMock(return_value=(mock_curate_response, 250, 60))):
            summary = await handlers.trigger_curation()

        assert summary["added_facts"] == 1
        all_mem = memory.get_all_memories()
        assert len(all_mem["memories"]) == 1
        assert all_mem["memories"][0]["topic"] == "Bakery"
        assert os.path.exists(mf)
        print("Smoke test successfully executed full conversational and memory cycle.")
