import asyncio
import logging
from datetime import datetime
import html
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from handlers import BotHandlers
from llm import LLMClient
from memory import MemoryManager
from params import Params
from state import StateManager
from telegram import ReactionTypeEmoji


@pytest.fixture
def test_setup():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        mf = os.path.join(td, "mem.json")
        p = Params()
        p.bot_token = "123:ABC"
        p.group_chat_id = -1001234567890
        p.admin_user_ids = [999, 888]
        p.model_api_key = "test-key"

        s = StateManager(sf, group_chat_id=-1001234567890)
        s.load()

        m = MemoryManager(mf)
        m.load()

        llm = LLMClient(p, s)
        handlers = BotHandlers(p, s, m, llm)
        yield p, s, m, llm, handlers


def test_authorization_checks(test_setup):
    p, s, m, llm, handlers = test_setup
    assert handlers._is_admin(999) is True
    assert handlers._is_admin(888) is True
    assert handlers._is_admin(111) is False
    assert handlers._is_admin(None) is False


@pytest.mark.asyncio
async def test_unauthorized_group_leave(test_setup):
    p, s, m, llm, handlers = test_setup

    mock_chat = MagicMock()
    mock_chat.id = -1009999999999  # Unauthorized group ID
    mock_chat.type = "supergroup"
    mock_chat.title = "Unauthorized Group"

    mock_update = MagicMock()
    mock_update.effective_chat = mock_chat
    mock_update.effective_message = MagicMock(migrate_to_chat_id=None)

    mock_context = MagicMock()
    mock_context.bot = MagicMock()
    mock_context.bot.leave_chat = AsyncMock()

    allowed = await handlers._check_group_authorization(mock_update, mock_context)
    assert allowed is False
    mock_context.bot.leave_chat.assert_awaited_once_with(-1009999999999)


@pytest.mark.asyncio
async def test_supergroup_migration(test_setup):
    p, s, m, llm, handlers = test_setup

    mock_chat = MagicMock()
    mock_chat.id = -1001234567890
    mock_chat.type = "group"

    mock_update = MagicMock()
    mock_update.effective_chat = mock_chat
    mock_update.effective_message = MagicMock(migrate_to_chat_id=-100555666777)

    mock_context = MagicMock()
    mock_context.bot = MagicMock()

    allowed = await handlers._check_group_authorization(mock_update, mock_context)
    assert allowed is True
    assert p.group_chat_id == -100555666777
    assert s.get_group_chat_id() == -100555666777


@pytest.mark.asyncio
async def test_supergroup_id_auto_normalization(test_setup):
    p, s, m, llm, handlers = test_setup
    p.group_chat_id = -1941958689
    s.set_group_chat_id(-1941958689)

    mock_chat = MagicMock()
    mock_chat.id = -1001941958689
    mock_chat.type = "supergroup"
    mock_chat.title = "Dirr"

    mock_update = MagicMock()
    mock_update.effective_chat = mock_chat
    mock_update.effective_message = MagicMock(migrate_to_chat_id=None)

    mock_context = MagicMock()
    mock_context.bot = MagicMock()

    allowed = await handlers._check_group_authorization(mock_update, mock_context)
    assert allowed is True
    assert p.group_chat_id == -1001941958689
    assert s.get_group_chat_id() == -1001941958689
    mock_context.bot.leave_chat.assert_not_called()

def test_is_direct_trigger(test_setup):
    p, s, m, llm, handlers = test_setup

    # 1. Mentioning @pletykas_bot
    msg_mention = MagicMock(reply_to_message=None, text="Hello @pletykas_bot, what's up?", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_mention, bot_username="pletykas_bot") is True

    # 2. Mentioning pletykas_bot without @
    msg_no_at = MagicMock(reply_to_message=None, text="pletykas_bot: what do you think?", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_no_at, bot_username="pletykas_bot") is True

    # 3. Mentioning bot by name (e.g. Pletykas, Pletykás)
    msg_name = MagicMock(reply_to_message=None, text="Szia Pletykás, mit gondolsz?", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_name, bot_username="pletykas_bot", bot_name="Pletykas") is True

    # Default nicknames (pletyi, pletyo)
    msg_def_nick = MagicMock(reply_to_message=None, text="pletyi, mi a helyzet?", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_def_nick, bot_username="pletykas_bot") is True

    # 4. Configured nicknames
    s.set_nicknames(["pleti", "kis pletyka"])
    msg_nick1 = MagicMock(reply_to_message=None, text="pleti, hallottad ezt?", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_nick1, bot_username="pletykas_bot") is True

    msg_nick_at = MagicMock(reply_to_message=None, text="Szia @pleti!", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_nick_at, bot_username="pletykas_bot") is True

    msg_multiword = MagicMock(reply_to_message=None, text="Te kis pletyka, mit csinálsz?", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_multiword, bot_username="pletykas_bot") is True

    # 5. Negative word boundary checks
    msg_neg1 = MagicMock(reply_to_message=None, text="Ez komplett hülyeség", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_neg1, bot_username="pletykas_bot") is False

    msg_neg2 = MagicMock(reply_to_message=None, text="napletykas_bot teszt", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_neg2, bot_username="pletykas_bot") is False

    msg_neg3 = MagicMock(reply_to_message=None, text="pletykas_bot_2 teszt", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_neg3, bot_username="pletykas_bot") is False

    # 6. Replying to bot's message
    msg_reply = MagicMock(text="Tell me more", caption=None, entities=[])
    bot_user = MagicMock(is_bot=True, id=9999, username="pletykas_bot")
    msg_reply.reply_to_message = MagicMock(from_user=bot_user)
    assert handlers._is_direct_trigger(msg_reply, bot_username="pletykas_bot", bot_id=9999) is True

    # 6b. Bot display name with extra words (e.g. "Pletykas Bot") matches single word "Pletykas"
    msg_bot_first_name = MagicMock(reply_to_message=None, text="Pletykas, mit gondolsz?", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_bot_first_name, bot_username="pletykas_bot", bot_name="Pletykas Bot") is True

    # 6c. Photo with caption mention entity
    mention_entity = MagicMock(type="mention", offset=0, length=13)
    msg_photo_caption = MagicMock(reply_to_message=None, text=None, caption="@pletykas_bot nézd", entities=[], caption_entities=[mention_entity])
    assert handlers._is_direct_trigger(msg_photo_caption, bot_username="pletykas_bot") is True

    # 7. Regular message between humans
    msg_human = MagicMock(reply_to_message=None, text="Hey Bob, want to grab lunch?", caption=None, entities=[])
    assert handlers._is_direct_trigger(msg_human, bot_username="pletykas_bot") is False

@pytest.mark.asyncio
async def test_admin_commands_private_chat(test_setup):
    p, s, m, llm, handlers = test_setup

    mock_msg = MagicMock()
    mock_msg.reply_text = AsyncMock()

    mock_update = MagicMock()
    mock_update.effective_user.id = 999
    mock_update.effective_chat.type = "private"
    mock_update.effective_message = mock_msg

    mock_context = MagicMock()

    # /status
    mock_msg.reply_text.reset_mock()
    await handlers.cmd_status(mock_update, mock_context)
    status_reply = mock_msg.reply_text.call_args[0][0]
    assert "<b>Pletykas Status</b>" in status_reply
    assert "Observed Messages" not in status_reply
    assert "Bot Replies" not in status_reply
    assert "Last Reply" not in status_reply
    assert "• Stats:\n  - Uncurated Messages:" in status_reply

    # /talkativeness
    mock_context.args = ["8"]
    await handlers.cmd_talkativeness(mock_update, mock_context)
    assert s.get_talkativeness() == 8
    mock_msg.reply_text.assert_awaited()

    # /cooldown
    mock_context.args = ["10"]
    await handlers.cmd_cooldown(mock_update, mock_context)
    assert s.get_cooldown_sec() == 10

    # /language
    mock_context.args = ["English"]
    await handlers.cmd_language(mock_update, mock_context)
    assert s.get_language() == "English"

    # /timezone
    mock_context.args = ["America/New_York"]
    await handlers.cmd_timezone(mock_update, mock_context)
    assert s.get_timezone() == "America/New_York"

    # /nicknames view default
    mock_msg.text = "/nicknames"
    await handlers.cmd_nicknames(mock_update, mock_context)
    assert "pletyi, pletyo" in mock_msg.reply_text.call_args[0][0]

    # /nicknames set
    mock_msg.text = "/nicknames pletyka, @pleti, kis pletyka"
    await handlers.cmd_nicknames(mock_update, mock_context)
    assert s.get_nicknames() == ["pletyka", "pleti", "kis pletyka"]
    assert "Updated nicknames" in mock_msg.reply_text.call_args[0][0]

    # /nicknames view configured
    mock_msg.text = "/nicknames"
    await handlers.cmd_nicknames(mock_update, mock_context)
    assert "pletyka, pleti, kis pletyka" in mock_msg.reply_text.call_args[0][0]

    # /nicknames clear
    mock_msg.text = "/nicknames clear"
    await handlers.cmd_nicknames(mock_update, mock_context)
    assert s.get_nicknames() == []
    assert "Cleared all nicknames" in mock_msg.reply_text.call_args[0][0]

    # /nicknames view after clear
    mock_msg.text = "/nicknames"
    await handlers.cmd_nicknames(mock_update, mock_context)
    assert "No nicknames configured" in mock_msg.reply_text.call_args[0][0]

    # /sleep
    mock_context.args = ["on", "22:00", "06:00"]
    await handlers.cmd_sleep(mock_update, mock_context)
    sched = s.get_sleep_schedule()
    assert sched["enabled"] is True
    assert sched["sleep_start"] == "22:00"

    # /spontaneous
    mock_context.args = ["on"]
    await handlers.cmd_spontaneous(mock_update, mock_context)
    assert s.is_spontaneous_enabled() is True
    mock_context.job_queue.run_once.assert_called()

    mock_context.args = ["off"]
    await handlers.cmd_spontaneous(mock_update, mock_context)
    assert s.is_spontaneous_enabled() is False

    # /spontaneous_interval
    mock_context.args = ["3", "5"]
    await handlers.cmd_spontaneous_interval(mock_update, mock_context)
    assert s.get_spontaneous_settings()["min_hours"] == 3.0
    assert s.get_spontaneous_settings()["max_hours"] == 5.0

    # /spontaneous shortcut with numbers: /spontaneous 1.5 3.5
    mock_context.args = ["1.5", "3.5"]
    await handlers.cmd_spontaneous(mock_update, mock_context)
    assert s.get_spontaneous_settings()["min_hours"] == 1.5
    assert s.get_spontaneous_settings()["max_hours"] == 3.5

    # /spontaneous_now (text message)
    with patch.object(llm, "generate_spontaneous_message", AsyncMock(return_value="Hey everyone, did you know that honey never spoils?")):
        mock_context.bot.send_message = AsyncMock(return_value=MagicMock(message_id=5001))
        await handlers.cmd_spontaneous_now(mock_update, mock_context)
        mock_context.bot.send_message.assert_awaited()
        call_kwargs = mock_context.bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == p.group_chat_id
        assert "honey never spoils" in call_kwargs["text"]

    # /spontaneous now subcommand (poll)
    poll_llm_out = "<POLL>\nQuestion: Best programming language?\n- Python\n- Rust\n- TypeScript\n</POLL>"
    with patch.object(llm, "generate_spontaneous_message", AsyncMock(return_value=poll_llm_out)):
        mock_context.bot.send_poll = AsyncMock(return_value=MagicMock(message_id=5002))
        mock_context.args = ["now"]
        await handlers.cmd_spontaneous(mock_update, mock_context)
        mock_context.bot.send_poll.assert_awaited()
        call_kwargs = mock_context.bot.send_poll.call_args[1]
        assert call_kwargs["chat_id"] == p.group_chat_id
        assert call_kwargs["question"] == "Best programming language?"
        assert len(call_kwargs["options"]) == 3

    # /grounding
    mock_context.args = ["on"]
    await handlers.cmd_grounding(mock_update, mock_context)
    assert s.is_search_grounding_active() is True

    # /debug
    mock_context.args = []
    await handlers.cmd_debug(mock_update, mock_context)
    assert "Debug mode is currently" in mock_msg.reply_text.call_args[0][0]

    mock_context.args = ["on"]
    await handlers.cmd_debug(mock_update, mock_context)
    assert s.is_debug_mode() is True

    mock_context.args = ["off"]
    await handlers.cmd_debug(mock_update, mock_context)
    assert s.is_debug_mode() is False

    # /image_large
    mock_context.args = []
    await handlers.cmd_image_large_model(mock_update, mock_context)
    assert "Large Model for Image Interpretation is currently" in mock_msg.reply_text.call_args[0][0]

    mock_context.args = ["off"]
    await handlers.cmd_image_large_model(mock_update, mock_context)
    assert s.is_image_interpretation_large_model() is False

    mock_context.args = ["on"]
    await handlers.cmd_image_large_model(mock_update, mock_context)
    assert s.is_image_interpretation_large_model() is True


    # /search_small
    mock_context.args = []
    await handlers.cmd_search_small(mock_update, mock_context)
    assert "Small Model Web Search is currently" in mock_msg.reply_text.call_args[0][0]

    mock_context.args = ["on"]
    await handlers.cmd_search_small(mock_update, mock_context)
    assert s.is_search_small_model() is True

    mock_context.args = ["off"]
    await handlers.cmd_search_small(mock_update, mock_context)
    assert s.is_search_small_model() is False
    # /prompt view (downloads prompt document)
    mock_msg.document = None
    mock_msg.reply_document = AsyncMock()
    mock_context.args = []
    await handlers.cmd_prompt(mock_update, mock_context)
    mock_msg.reply_document.assert_awaited_once()

    # /prompt reset
    mock_context.args = ["reset"]
    s.set_system_prompt("Custom prompt before reset")
    await handlers.cmd_prompt(mock_update, mock_context)
    assert s.get_system_prompt() != "Custom prompt before reset"

    # /prompt load
    mock_context.args = ["load"]
    await handlers.cmd_prompt(mock_update, mock_context)
    assert mock_update.effective_user.id in handlers._waiting_for_prompt_upload
    mock_msg.reply_text.assert_awaited()

    # /cancel
    mock_msg.reply_text.reset_mock()
    await handlers.cmd_cancel(mock_update, mock_context)
    assert mock_update.effective_user.id not in handlers._waiting_for_prompt_upload
    assert "cancelled" in mock_msg.reply_text.call_args[0][0].lower()
    # /memories (upload json as-is)
    mock_msg.document = None
    mock_msg.reply_document = AsyncMock()
    mock_context.args = []
    await handlers.cmd_memories(mock_update, mock_context)
    mock_msg.reply_document.assert_awaited_once()

    # /memories load
    mock_context.args = ["load"]
    await handlers.cmd_memories(mock_update, mock_context)
    assert mock_update.effective_user.id in handlers._waiting_for_memory_upload
    mock_msg.reply_text.assert_awaited()

    # /cancel
    mock_msg.reply_text.reset_mock()
    await handlers.cmd_cancel(mock_update, mock_context)
    assert mock_update.effective_user.id not in handlers._waiting_for_memory_upload
    assert "cancelled" in mock_msg.reply_text.call_args[0][0].lower()
@pytest.mark.asyncio
async def test_debounce_job_handling(test_setup):
    p, s, m, llm, handlers = test_setup
    s.set_spontaneous_settings(enabled=False)

    chat_id = -1001234567890
    mock_job1 = MagicMock()
    mock_job1.schedule_removal = MagicMock()
    handlers._debounce_jobs[chat_id] = mock_job1

    # Incoming message simulation with context.job_queue
    mock_msg = MagicMock()
    mock_msg.message_id = 1001
    mock_msg.from_user.id = 555
    mock_msg.from_user.full_name = "User"
    mock_msg.text = "Hello team"
    mock_msg.caption = None
    mock_msg.photo = []
    mock_msg.voice = None
    mock_msg.audio = None
    mock_msg.video = None
    mock_msg.video_note = None
    mock_msg.document = None
    mock_msg.sticker = None
    mock_msg.animation = None
    mock_msg.reply_to_message = None
    mock_msg.entities = []
    mock_msg.migrate_to_chat_id = None
    mock_msg.date = None

    mock_chat = MagicMock(id=chat_id, type="supergroup")
    mock_update = MagicMock(effective_chat=mock_chat, effective_message=mock_msg, effective_user=mock_msg.from_user)

    mock_context = MagicMock()
    mock_context.bot.id = 999999
    mock_context.bot.username = "pletykas_bot"
    mock_context.job_queue.run_once = MagicMock(return_value=MagicMock())

    await handlers.on_message(mock_update, mock_context)

    # Old job must be cancelled
    mock_job1.schedule_removal.assert_called_once()
    # New job scheduled via run_once
    mock_context.job_queue.run_once.assert_called_once()
    call_kwargs = mock_context.job_queue.run_once.call_args[1]
    assert call_kwargs["when"] == 5
    assert "job_kwargs" in call_kwargs
    assert call_kwargs["job_kwargs"]["id"].startswith(f"debounce_{chat_id}_")
    assert len(s.get_chat_history()) == 1

@pytest.mark.asyncio
async def test_bot_mention_or_nickname_ignores_cooldown_and_evaluates_immediately(test_setup):
    p, s, m, llm, handlers = test_setup
    s.set_spontaneous_settings(enabled=False)
    assert s.get_cooldown_sec() == 5

    chat_id = -1001234567890
    # Pending debounce job from an earlier message
    mock_job1 = MagicMock()
    mock_job1.schedule_removal = MagicMock()
    handlers._debounce_jobs[chat_id] = mock_job1

    mock_msg = MagicMock()
    mock_msg.message_id = 2001
    mock_msg.from_user.id = 555
    mock_msg.from_user.full_name = "User"
    mock_msg.text = "pletyi, mondj egy viccet!"
    mock_msg.caption = None
    mock_msg.photo = []
    mock_msg.voice = None
    mock_msg.audio = None
    mock_msg.video = None
    mock_msg.video_note = None
    mock_msg.document = None
    mock_msg.sticker = None
    mock_msg.animation = None
    mock_msg.reply_to_message = None
    mock_msg.entities = []
    mock_msg.migrate_to_chat_id = None
    mock_msg.date = None

    mock_chat = MagicMock(id=chat_id, type="supergroup")
    mock_update = MagicMock(effective_chat=mock_chat, effective_message=mock_msg, effective_user=mock_msg.from_user)

    mock_context = MagicMock()
    mock_context.bot.id = 999999
    mock_context.bot.username = "pletykas_bot"
    mock_context.bot.first_name = "Pletykas"
    mock_context.job_queue.run_once = MagicMock()

    with patch.object(handlers, "_execute_evaluation", AsyncMock()) as mock_eval:
        await handlers.on_message(mock_update, mock_context)

        # Old debounce job must be cancelled immediately
        mock_job1.schedule_removal.assert_called_once()
        assert chat_id not in handlers._debounce_jobs

        # Cooldown timer MUST NOT be scheduled (cooldown ignored)
        mock_context.job_queue.run_once.assert_not_called()

        # Evaluation must be started immediately with is_direct_trigger=True
        await asyncio.sleep(0)
        mock_eval.assert_awaited_once_with(
            context=mock_context,
            chat_id=chat_id,
            is_direct_trigger=True,
            trigger_msg_id=2001,
        )
    assert len(s.get_chat_history()) == 1

def test_debounce_job_logging_suppressed():
    from main import APSchedulerJobDurationFilter, DebounceJobFilter, HTTPRequestLogFilter

    http_filter = HTTPRequestLogFilter()
    rec_httpx = logging.LogRecord("httpx", logging.INFO, "client.py", 10, 'HTTP Request: POST https://api.telegram.org/bot123/sendMessage "HTTP/1.1 200 OK"', (), None)
    rec_app = logging.LogRecord("pletykas", logging.INFO, "handlers.py", 10, 'Dispatched message', (), None)
    assert http_filter.filter(rec_httpx) is False
    assert http_filter.filter(rec_app) is True

    f1 = DebounceJobFilter()
    f2 = APSchedulerJobDurationFilter()

    rec_debounce_run = logging.LogRecord(
        "apscheduler.executors.default",
        logging.INFO,
        "executors.py",
        10,
        'Running job "%s" (execution %d)',
        ('debounce_-1001234 (trigger: date[2026-09-25 12:00:00], next run at: 2026-09-25 12:00:00)', 1),
        None,
    )
    rec_debounce_success = logging.LogRecord(
        "apscheduler.executors.default",
        logging.INFO,
        "executors.py",
        20,
        'Job "%s" executed successfully',
        ('debounce_-1001234',),
        None,
    )
    rec_debounce_remove = logging.LogRecord(
        "apscheduler.scheduler",
        logging.INFO,
        "base.py",
        30,
        'Removed job %s',
        ('debounce_-1001234_abc123',),
        None,
    )
    assert f1.filter(rec_debounce_remove) is False
    assert f2.filter(rec_debounce_remove) is False
    rec_other = logging.LogRecord(
        "apscheduler.executors.default",
        logging.INFO,
        "executors.py",
        10,
        'Running job "%s" (execution %d)',
        ('spontaneous_revival', 1),
        None,
    )

    assert f1.filter(rec_debounce_run) is False
    assert f2.filter(rec_debounce_run) is False
    assert f1.filter(rec_debounce_success) is False
    assert f2.filter(rec_debounce_success) is False
    assert f1.filter(rec_other) is True
    assert f2.filter(rec_other) is True
def test_resolve_reaction_emoji():
    # Direct allowed emojis
    assert BotHandlers.resolve_reaction_emoji("🔥") == "🔥"
    assert BotHandlers.resolve_reaction_emoji("👍") == "👍"
    assert BotHandlers.resolve_reaction_emoji("😁") == "😁"
    assert BotHandlers.resolve_reaction_emoji("🤣") == "🤣"

    # Mapped emojis
    assert BotHandlers.resolve_reaction_emoji("😄") == "😁"
    assert BotHandlers.resolve_reaction_emoji("😂") == "🤣"
    assert BotHandlers.resolve_reaction_emoji("❤️") == "❤"
    assert BotHandlers.resolve_reaction_emoji("🥺") == "😢"
    assert BotHandlers.resolve_reaction_emoji("💀") == "👻"
    assert BotHandlers.resolve_reaction_emoji("😏") == "😈"
    assert BotHandlers.resolve_reaction_emoji("😉") == "😘"
    assert BotHandlers.resolve_reaction_emoji("🙄") == "🤨"
    # Unsupported emoji
    assert BotHandlers.resolve_reaction_emoji("🚀") is None

@pytest.mark.asyncio
async def test_reaction_dispatch_and_error_handling(test_setup):
    p, s, m, llm, handlers = test_setup

    mock_bot = MagicMock()
    mock_bot.id = 9999
    mock_bot.first_name = "Pletykas"
    mock_bot.set_message_reaction = AsyncMock()
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=2001))
    mock_context = MagicMock(bot=mock_bot)

    # 1. Mapped emoji: 😄 resolves to 😁
    eval_res = ("Nice message", ("😄", 1234), None, None)
    with patch.object(llm, "evaluate_and_reply", AsyncMock(return_value=eval_res)):
        await handlers._execute_evaluation(mock_context, p.group_chat_id, is_direct_trigger=True, trigger_msg_id=1234)
    mock_bot.set_message_reaction.assert_awaited_with(
        chat_id=p.group_chat_id,
        message_id=1234,
        reaction=[ReactionTypeEmoji(emoji="😁")],
    )

    # 2. Reaction_invalid error from Telegram API is caught gracefully
    mock_bot.set_message_reaction = AsyncMock(side_effect=Exception("Reaction_invalid"))
    with patch.object(llm, "evaluate_and_reply", AsyncMock(return_value=eval_res)):
        await handlers._execute_evaluation(mock_context, p.group_chat_id, is_direct_trigger=True, trigger_msg_id=1234)
    # Reply was still delivered even though reaction failed
    assert mock_bot.send_message.call_count >= 2


@pytest.mark.asyncio
async def test_trigger_spontaneous_message_errors(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_context = MagicMock()
    # No group id
    p.group_chat_id = 0
    ok, err = await handlers.trigger_spontaneous_message(mock_context)
    assert ok is False
    assert "Group chat ID is not configured" in err

    # LLM error
    p.group_chat_id = -100123
    with patch.object(llm, "generate_spontaneous_message", AsyncMock(side_effect=RuntimeError("API timeout"))):
        ok, err = await handlers.trigger_spontaneous_message(mock_context)
        assert ok is False
        assert "API timeout" in err

@pytest.mark.asyncio
async def test_spontaneous_random_timer_and_sleep_callback(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_context = MagicMock()
    mock_context.job_queue.get_jobs_by_name.return_value = []
    mock_context.job_queue.run_once = MagicMock()

    s.set_spontaneous_settings(enabled=True, min_hours=2.0, max_hours=4.0)

    # 1. schedule_spontaneous_job schedules within 2 to 4 hours (7200s to 14400s)
    handlers.schedule_spontaneous_job(mock_context)
    mock_context.job_queue.run_once.assert_called_once()
    when_arg = mock_context.job_queue.run_once.call_args[1]["when"]
    assert 7200.0 <= when_arg <= 14400.0

    # 2. _spontaneous_callback during sleep: does not trigger message, reschedules timer
    with patch.object(s, "is_sleeping", return_value=True):
        with patch.object(handlers, "trigger_spontaneous_message", AsyncMock()) as mock_trigger:
            with patch.object(handlers, "schedule_spontaneous_job") as mock_resched:
                await handlers._spontaneous_callback(mock_context)
                mock_trigger.assert_not_called()
                mock_resched.assert_called_once_with(mock_context)

    # 3. _spontaneous_callback while awake: triggers message and reschedules timer
    with patch.object(s, "is_sleeping", return_value=False):
        with patch.object(handlers, "trigger_spontaneous_message", AsyncMock()) as mock_trigger:
            with patch.object(handlers, "schedule_spontaneous_job") as mock_resched:
                await handlers._spontaneous_callback(mock_context)
                mock_trigger.assert_called_once_with(mock_context)
                mock_resched.assert_called_once_with(mock_context)

@pytest.mark.asyncio
async def test_status_cmd_shows_next_spontaneous_firing_timestamp(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_msg = AsyncMock()
    mock_update = MagicMock()
    mock_update.effective_user.id = 999
    mock_update.effective_chat.type = "private"
    mock_update.effective_message = mock_msg

    mock_context = MagicMock()
    mock_context.job_queue.get_jobs_by_name.return_value = []
    mock_context.job_queue.run_once = MagicMock()

    # 1. Spontaneous disabled -> Next: None
    s.set_spontaneous_settings(enabled=False)
    await handlers.cmd_status(mock_update, mock_context)
    status_reply = mock_msg.reply_text.call_args[0][0]
    assert "• Spontaneous Messages: <code>OFF (every 2-4h, random) (Next: None)</code>" in status_reply

    # 2. Spontaneous enabled & scheduled
    s.set_spontaneous_settings(enabled=True, min_hours=2.0, max_hours=4.0)
    handlers.schedule_spontaneous_job(mock_context)
    mock_msg.reply_text.reset_mock()
    await handlers.cmd_status(mock_update, mock_context)
    status_reply = mock_msg.reply_text.call_args[0][0]
    assert "• Spontaneous Messages: <code>ON (every 2-4h, random) (Next: 20" in status_reply
    assert "• Small Model Search: <code>" in status_reply
    assert "  - Larger: <code>" in status_reply

    # 3. /spontaneous with no args shows Next Firing timestamp
    mock_context.args = []
    mock_msg.reply_text.reset_mock()
    await handlers.cmd_spontaneous(mock_update, mock_context)
    spont_reply = mock_msg.reply_text.call_args[0][0]
    assert "🎲 Spontaneous revival messages: <b>ON</b>" in spont_reply
    assert "• Next Firing: <b>20" in spont_reply

    # 4. Job with next_t attribute from APScheduler is used directly
    specific_time = datetime(2026, 9, 25, 18, 30, 0, tzinfo=s.get_tzinfo())
    mock_job = MagicMock()
    mock_job.removed = False
    mock_job.next_t = specific_time
    mock_context.job_queue.get_jobs_by_name.return_value = [mock_job]
    mock_msg.reply_text.reset_mock()
    await handlers.cmd_status(mock_update, mock_context)
    status_reply = mock_msg.reply_text.call_args[0][0]
    assert "• Spontaneous Messages: <code>ON (every 2-4h, random) (Next: 2026-09-25 18:30:00)</code>" in status_reply

@pytest.mark.asyncio
async def test_spontaneous_job_persists_and_resumes_across_restarts(test_setup):
    from datetime import timedelta
    p, s, m, llm, handlers = test_setup
    mock_context = MagicMock()
    mock_context.job_queue.get_jobs_by_name.return_value = []
    mock_context.job_queue.run_once = MagicMock()

    s.set_spontaneous_settings(enabled=True, min_hours=2.0, max_hours=4.0)

    # 1. Initial scheduling persists next_fire_time to state
    handlers.schedule_spontaneous_job(mock_context)
    saved_fire_time = s.get_spontaneous_next_fire_time()
    assert saved_fire_time is not None
    assert saved_fire_time > datetime.now(s.get_tzinfo())

    # 2. Simulate restart: future timestamp is preserved, delay is remaining seconds
    future_time = datetime.now(s.get_tzinfo()) + timedelta(seconds=1800)
    s.set_spontaneous_next_fire_time(future_time)
    mock_context.job_queue.run_once.reset_mock()

    handlers.schedule_spontaneous_job(mock_context)
    mock_context.job_queue.run_once.assert_called_once()
    when_arg = mock_context.job_queue.run_once.call_args[1]["when"]
    # when_arg should be approximately 1800 seconds
    assert 1790 <= when_arg <= 1805
    # The persisted fire time should NOT have been overwritten with a new random interval
    assert s.get_spontaneous_next_fire_time() == future_time

    # 3. Simulate restart with past timestamp: schedules a new one
    past_time = datetime.now(s.get_tzinfo()) - timedelta(seconds=600)
    s.set_spontaneous_next_fire_time(past_time)
    mock_context.job_queue.run_once.reset_mock()

    handlers.schedule_spontaneous_job(mock_context)
    mock_context.job_queue.run_once.assert_called_once()
    when_arg = mock_context.job_queue.run_once.call_args[1]["when"]
    # New random interval between 2h and 4h (7200s to 14400s)
    assert 7200 <= when_arg <= 14400
    # A new future fire time is persisted
    new_fire_time = s.get_spontaneous_next_fire_time()
    assert new_fire_time is not None
    assert new_fire_time > datetime.now(s.get_tzinfo())
    assert new_fire_time != past_time

@pytest.mark.asyncio
async def test_group_message_reschedules_spontaneous_job(test_setup):
    from datetime import timedelta
    p, s, m, llm, handlers = test_setup
    p.group_chat_id = -1001234567890

    mock_chat = MagicMock()
    mock_chat.id = -1001234567890
    mock_chat.type = "supergroup"

    mock_user = MagicMock()
    mock_user.id = 111111
    mock_user.full_name = "Alice"

    mock_msg = MagicMock()
    mock_msg.message_id = 5001
    mock_msg.from_user = mock_user
    mock_msg.text = "Hello everyone in the group!"
    mock_msg.photo = None
    mock_msg.voice = None
    mock_msg.audio = None
    mock_msg.video = None
    mock_msg.video_note = None
    mock_msg.document = None
    mock_msg.sticker = None
    mock_msg.animation = None
    mock_msg.caption = None
    mock_msg.reply_to_message = None

    mock_update = MagicMock(effective_chat=mock_chat, effective_message=mock_msg, effective_user=mock_user)

    mock_context = MagicMock()
    mock_context.bot.id = 999999
    mock_context.bot.username = "pletykas_bot"
    mock_context.bot.first_name = "Pletykas"
    mock_context.job_queue.get_jobs_by_name.return_value = []
    mock_context.job_queue.run_once = MagicMock(return_value=MagicMock())

    # 1. Enable spontaneous messages and set an initial scheduled future timestamp
    s.set_spontaneous_settings(enabled=True, min_hours=2.0, max_hours=4.0)
    old_future_time = datetime.now(s.get_tzinfo()) + timedelta(seconds=3600)
    s.set_spontaneous_next_fire_time(old_future_time)
    handlers._next_spontaneous_time = old_future_time

    # 2. When a telegram message arrives in the group, spontaneous job should reschedule with a new timestamp
    with patch.object(handlers, "schedule_spontaneous_job", wraps=handlers.schedule_spontaneous_job) as spy_resched:
        await handlers.on_message(mock_update, mock_context)
        spy_resched.assert_called_once_with(mock_context, force_new=True)

    # The persisted timestamp must be updated and different from the old one
    new_fire_time = s.get_spontaneous_next_fire_time()
    assert new_fire_time is not None
    assert new_fire_time != old_future_time
    assert new_fire_time > datetime.now(s.get_tzinfo())

    # 3. When spontaneous messages are disabled, on_message should not reschedule
    s.set_spontaneous_settings(enabled=False)
    with patch.object(handlers, "schedule_spontaneous_job", wraps=handlers.schedule_spontaneous_job) as spy_resched:
        mock_msg.message_id = 5002
        await handlers.on_message(mock_update, mock_context)
        spy_resched.assert_not_called()
@pytest.mark.asyncio
async def test_photo_reply_and_reference_triggers_vision(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_bot = MagicMock()
    mock_bot.id = 9999
    mock_bot.first_name = "Pletykas"
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=3001))
    mock_context = MagicMock(bot=mock_bot)

    import base64
    fake_b64 = base64.b64encode(b"fake-photo-data").decode("utf-8")

    # Insert an earlier photo message into history
    photo_msg = {
        "id": 100,
        "from_user_id": 1,
        "from_user_name": "User",
        "text": "[Photo]",
        "media_type": "photo",
        "media_b64": fake_b64,
    }
    s.append_chat_message(photo_msg)

    # 1. User sends a follow-up text asking about the picture: "szoval mi van a kepen?"
    text_msg = {
        "id": 101,
        "from_user_id": 1,
        "from_user_name": "User",
        "text": "szoval mi van a kepen?",
        "media_type": "none",
    }
    s.append_chat_message(text_msg)

    with patch.object(llm, "describe_and_reply_image", AsyncMock(return_value=("Két cica van a képen!", None, None, "Két cica", None))) as mock_vision:
        await handlers._execute_evaluation(mock_context, p.group_chat_id, is_direct_trigger=True, trigger_msg_id=101)
        mock_vision.assert_awaited_once()
        kwargs = mock_vision.call_args.kwargs
        assert kwargs["image_bytes"] == b"fake-photo-data"
        assert kwargs["caption"] == "szoval mi van a kepen?"
        mock_bot.send_message.assert_awaited()
        assert "Két cica van a képen!" in mock_bot.send_message.call_args.kwargs["text"]

@pytest.mark.asyncio
async def test_image_modify_via_multimodal_vision(test_setup):
    import base64
    from PIL import Image
    import io
    p, s, m, llm, handlers = test_setup

    # Create valid dummy PNG image bytes
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    fake_img_bytes = buf.getvalue()

    mock_bot = MagicMock()
    mock_bot.id = 9999
    mock_bot.first_name = "Pletykas"
    mock_bot.set_message_reaction = AsyncMock()
    mock_bot.send_photo = AsyncMock(return_value=MagicMock(message_id=63614))
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=63615))
    mock_bot.send_chat_action = AsyncMock()
    mock_context = MagicMock(bot=mock_bot)

    # 1. Historical photo message
    photo_msg = {
        "id": 63612,
        "from_user_id": 133687316,
        "from_user_name": "Norbert",
        "text": "[Photo: RoboCop action figure]",
        "media_type": "photo",
        "media_b64": base64.b64encode(fake_img_bytes).decode("utf-8"),
    }
    s.append_chat_message(photo_msg)

    # 2. User modification request
    req_msg = {
        "id": 63613,
        "from_user_id": 133687316,
        "from_user_name": "Norbert",
        "text": "pletyi, szerkeszd a kepet, vigyorogjon a robocop es legyen haja",
        "media_type": "none",
    }
    s.append_chat_message(req_msg)

    vision_eval = (
        None,
        ("😈", None),
        {
            "prompt": "Smiling RoboCop with hair",
            "caption": "Itt a vigyorgó robocop!",
            "source": "reply",
            "mode": "modify",
        },
        "RoboCop action figure",
        None,
    )

    with patch.object(llm, "describe_and_reply_image", AsyncMock(return_value=vision_eval)), \
         patch.object(llm, "generate_image", AsyncMock(return_value=fake_img_bytes)) as mock_gen:
        await handlers._execute_evaluation(mock_context, p.group_chat_id, is_direct_trigger=True, trigger_msg_id=63613)

        # Verify reaction was dispatched
        mock_bot.set_message_reaction.assert_awaited_once()
        assert mock_bot.set_message_reaction.call_args.kwargs["reaction"][0].emoji == "😈"

        # Verify generate_image was called with prompt and base image bytes
        mock_gen.assert_awaited_once()
        assert mock_gen.call_args.kwargs["prompt"] == "Smiling RoboCop with hair"
        assert mock_gen.call_args.kwargs["base_image_bytes"] == fake_img_bytes

        # Verify send_photo was called with the generated photo and caption
        mock_bot.send_photo.assert_awaited_once()
        photo_kwargs = mock_bot.send_photo.call_args.kwargs
        assert photo_kwargs["chat_id"] == p.group_chat_id
        assert photo_kwargs["caption"] == "Itt a vigyorgó robocop!"
@pytest.mark.asyncio
async def test_debug_mode_logs_group_messages(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_bot = MagicMock()
    mock_bot.id = 9999
    mock_bot.first_name = "Pletykas"
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=4001))
    mock_context = MagicMock(bot=mock_bot)

    mock_update = MagicMock()
    mock_chat = MagicMock()
    mock_chat.id = p.group_chat_id
    mock_chat.type = "supergroup"
    mock_msg = MagicMock()
    mock_msg.message_id = 12345
    mock_msg.text = "Hello group chat!"
    mock_msg.caption = None
    mock_msg.photo = None
    mock_msg.voice = None
    mock_msg.audio = None
    mock_msg.video = None
    mock_msg.video_note = None
    mock_msg.document = None
    mock_msg.sticker = None
    mock_msg.animation = None
    mock_msg.reply_to_message = None
    mock_msg.entities = []
    mock_msg.from_user = MagicMock(id=42, full_name="Alice")
    mock_update.effective_chat = mock_chat
    mock_update.effective_message = mock_msg

    with patch.object(handlers, "_log_debug_group_msg") as mock_log_dbg:
        # Debug mode OFF: nothing logged
        s.set_debug_mode(False)
        await handlers.on_message(mock_update, mock_context)
        mock_log_dbg.assert_not_called()

        # Debug mode ON: incoming group message logged
        s.set_debug_mode(True)
        mock_msg.message_id = 12346
        await handlers.on_message(mock_update, mock_context)
        assert mock_log_dbg.call_count >= 1
        inc_call = mock_log_dbg.call_args_list[0]
        assert inc_call[0][0] == "INCOMING"
        assert "12346" in inc_call[0][1]
        assert "Alice" in inc_call[0][1]
        assert "Hello group chat!" in inc_call[0][1]

        # Outgoing message logged
        mock_log_dbg.reset_mock()
        with patch.object(llm, "evaluate_and_reply", AsyncMock(return_value=("Hi Alice!", None, None, None))):
            await handlers._execute_evaluation(mock_context, p.group_chat_id, is_direct_trigger=True, trigger_msg_id=12346)
            assert mock_log_dbg.call_count >= 1
            out_call = mock_log_dbg.call_args_list[0]
            assert out_call[0][0] == "OUTGOING"
            assert "Hi Alice!" in out_call[0][1]

@pytest.mark.asyncio
async def test_cmd_memories_and_load_workflow(test_setup):
    import json
    p, s, m, llm, handlers = test_setup
    mock_update = MagicMock()
    mock_chat = MagicMock(type="private")
    mock_user = MagicMock(id=p.admin_user_ids[0])
    mock_msg = MagicMock()
    mock_msg.document = None
    mock_msg.reply_text = AsyncMock()
    mock_msg.reply_document = AsyncMock()
    mock_update.effective_chat = mock_chat
    mock_update.effective_user = mock_user
    mock_update.effective_message = mock_msg
    mock_context = MagicMock()

    # 1. /memories uploads JSON file as document
    mock_context.args = []
    await handlers.cmd_memories(mock_update, mock_context)
    mock_msg.reply_document.assert_awaited_once()

    # 2. /memories load prompts user and registers waiting state
    mock_context.args = ["load"]
    await handlers.cmd_memories(mock_update, mock_context)
    assert mock_user.id in handlers._waiting_for_memory_upload

    # 3. User sends valid JSON file document
    new_mem_data = {
        "version": 1,
        "memories": [
            {"topic": "Dan", "content": "DevOps expert"}
        ],
        "dynamics": [],
        "inside_jokes": []
    }
    raw_bytes = json.dumps(new_mem_data).encode("utf-8")

    mock_doc = MagicMock()
    mock_doc.file_id = "doc123"
    mock_doc.file_name = "new_memories.json"
    mock_doc.mime_type = "application/json"

    mock_file = MagicMock()
    async def fake_download(buf):
        buf.write(raw_bytes)
    mock_file.download_to_memory = fake_download
    mock_context.bot.get_file = AsyncMock(return_value=mock_file)

    doc_msg = MagicMock()
    doc_msg.document = mock_doc
    doc_msg.text = None
    doc_msg.caption = None
    doc_msg.reply_text = AsyncMock()
    doc_update = MagicMock()
    doc_update.effective_chat = mock_chat
    doc_update.effective_user = mock_user
    doc_update.effective_message = doc_msg

    await handlers.on_message(doc_update, mock_context)

    # Verify memory was updated and user is no longer in waiting state
    assert mock_user.id not in handlers._waiting_for_memory_upload
    assert len(m.get_all_memories()["memories"]) == 1
    assert m.get_all_memories()["memories"][0]["topic"] == "Dan"
    doc_msg.reply_text.assert_awaited()
    assert "Successfully" in doc_msg.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_cmd_prompt_and_load_workflow(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_update = MagicMock()
    mock_chat = MagicMock(type="private")
    mock_user = MagicMock(id=p.admin_user_ids[0])
    mock_msg = MagicMock()
    mock_msg.document = None
    mock_msg.reply_text = AsyncMock()
    mock_msg.reply_document = AsyncMock()
    mock_update.effective_chat = mock_chat
    mock_update.effective_user = mock_user
    mock_update.effective_message = mock_msg
    mock_context = MagicMock()

    # 1. /prompt uploads prompt file as document
    mock_context.args = []
    await handlers.cmd_prompt(mock_update, mock_context)
    mock_msg.reply_document.assert_awaited_once()
    doc_call_kwargs = mock_msg.reply_document.call_args[1]
    assert doc_call_kwargs["filename"] == "prompt.txt"

    # 2. /prompt load prompts user and registers waiting state
    mock_context.args = ["load"]
    await handlers.cmd_prompt(mock_update, mock_context)
    assert mock_user.id in handlers._waiting_for_prompt_upload

    # 3. User sends valid prompt text file document
    new_prompt_text = "You are a friendly Hungarian tavern gossip."
    raw_bytes = new_prompt_text.encode("utf-8")

    mock_doc = MagicMock()
    mock_doc.file_id = "prompt_doc_1"
    mock_doc.file_name = "new_prompt.txt"
    mock_doc.mime_type = "text/plain"

    mock_file = MagicMock()
    async def fake_download(buf):
        buf.write(raw_bytes)
    mock_file.download_to_memory = fake_download
    mock_context.bot.get_file = AsyncMock(return_value=mock_file)

    doc_msg = MagicMock()
    doc_msg.document = mock_doc
    doc_msg.text = None
    doc_msg.caption = None
    doc_msg.reply_text = AsyncMock()
    doc_update = MagicMock()
    doc_update.effective_chat = mock_chat
    doc_update.effective_user = mock_user
    doc_update.effective_message = doc_msg

    await handlers.on_message(doc_update, mock_context)

    # Verify prompt was updated and user is no longer in waiting state
    assert mock_user.id not in handlers._waiting_for_prompt_upload
    assert s.get_system_prompt() == new_prompt_text
    doc_msg.reply_text.assert_awaited()
    assert "Successfully" in doc_msg.reply_text.call_args[0][0]

    # 4. User sends empty prompt file
    handlers._waiting_for_prompt_upload.add(mock_user.id)
    empty_file = MagicMock()
    async def fake_empty_download(buf):
        buf.write(b"   \n  ")
    empty_file.download_to_memory = fake_empty_download
    mock_context.bot.get_file = AsyncMock(return_value=empty_file)
    doc_msg.reply_text.reset_mock()

    await handlers.on_message(doc_update, mock_context)
    assert "empty" in doc_msg.reply_text.call_args[0][0].lower()
    assert s.get_system_prompt() == new_prompt_text
def test_split_into_html_pre_chunks():
    from handlers import split_into_html_pre_chunks
    # Single short chunk
    chunks = split_into_html_pre_chunks("Short text", header="Header")
    assert len(chunks) == 1
    assert chunks[0] == "Header\n<pre>Short text</pre>"

    # Long text with special characters
    long_text = "Line with <b> & \"quotes\" and <tags>\n" * 150
    chunks = split_into_html_pre_chunks(long_text, header="Header", max_escaped_len=1000)
    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert len(c) <= 1200
        assert f"Header (Part {idx + 1}/{len(chunks)}):" in c
        assert "<pre>" in c and "</pre>" in c
        assert "&lt;b&gt;" in c



def test_parse_interval_relativedelta_and_helpers():
    from handlers import (
        parse_interval_relativedelta,
        serialize_interval,
        deserialize_interval,
        parse_schedule_time,
    )
    from dateutil.relativedelta import relativedelta
    import zoneinfo

    # Single units
    assert parse_interval_relativedelta("2h") == relativedelta(hours=2)
    assert parse_interval_relativedelta("3 days") == relativedelta(days=3)
    assert parse_interval_relativedelta("1 month") == relativedelta(months=1)
    assert parse_interval_relativedelta("1 year") == relativedelta(years=1)
    assert parse_interval_relativedelta("30 mins") == relativedelta(minutes=30)
    assert parse_interval_relativedelta("2 weeks") == relativedelta(days=14)

    # Compound units
    compound = parse_interval_relativedelta("1 month 2 days 3 hours")
    assert compound == relativedelta(months=1, days=2, hours=3)

    # Clamping < 60s by default (for periodic schedules)
    clamped = parse_interval_relativedelta("10s")
    assert clamped == relativedelta(seconds=60)
    clamped20 = parse_interval_relativedelta("20s")
    assert clamped20 == relativedelta(seconds=60)

    # unclamped with min_seconds=1 (for oneshot schedules)
    unclamped = parse_interval_relativedelta("20s", min_seconds=1)
    assert unclamped == relativedelta(seconds=20)
    # Invalid unit returns None
    assert parse_interval_relativedelta("blabla") is None
    assert parse_interval_relativedelta("") is None

    # Serialization / Deserialization
    rd = relativedelta(years=1, months=2, days=3, hours=4, minutes=5, seconds=6)
    serialized = serialize_interval(rd)
    assert serialized == {"years": 1, "months": 2, "days": 3, "hours": 4, "minutes": 5, "seconds": 6}
    deserialized = deserialize_interval(serialized)
    assert deserialized == rd

    # parse_schedule_time
    tz = zoneinfo.ZoneInfo("UTC")
    now = datetime(2026, 9, 26, 12, 0, 0, tzinfo=tz)

    # Relative offset
    res_rel = parse_schedule_time("in 2h", tz, now)
    assert res_rel == datetime(2026, 9, 26, 14, 0, 0, tzinfo=tz)
    res_rel2 = parse_schedule_time("+3 days", tz, now)
    assert res_rel2 == datetime(2026, 9, 29, 12, 0, 0, tzinfo=tz)
    res_rel3 = parse_schedule_time("+20s", tz, now)
    assert res_rel3 == datetime(2026, 9, 26, 12, 0, 20, tzinfo=tz)
    # Time-only (later today)
    res_time = parse_schedule_time("18:30", tz, now)
    assert res_time == datetime(2026, 9, 26, 18, 30, 0, tzinfo=tz)

    # Time-only (already passed today -> next day)
    res_time_past = parse_schedule_time("09:00", tz, now)
    assert res_time_past == datetime(2026, 9, 27, 9, 0, 0, tzinfo=tz)

    # Absolute ISO timestamp
    res_iso = parse_schedule_time("2026-10-01T10:00:00", tz, now)
    assert res_iso == datetime(2026, 10, 1, 10, 0, 0, tzinfo=tz)

    # Invalid
    assert parse_schedule_time("invalid_date_xyz", tz, now) is None


@pytest.mark.asyncio
async def test_execute_evaluation_with_schedule_actions(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_bot = MagicMock()
    mock_bot.id = 9999
    mock_bot.first_name = "Pletykas"
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=2001))
    mock_context = MagicMock(bot=mock_bot, job_queue=MagicMock())

    # 1. LLM outputs oneshot schedule creation
    sched_create_spec = {
        "action": "create",
        "type": "oneshot",
        "time": "+1h",
        "interval": "",
        "start": "",
        "description": "Remind Alice about meeting",
    }
    eval_res = ("Rendben, észben tartom!", None, None, sched_create_spec)

    with patch.object(llm, "evaluate_and_reply", AsyncMock(return_value=eval_res)):
        await handlers._execute_evaluation(mock_context, p.group_chat_id, is_direct_trigger=True, trigger_msg_id=1234)

    replies = s.get_scheduled_replies()
    assert len(replies) == 1
    created_id = replies[0]["id"]
    assert replies[0]["description"] == "Remind Alice about meeting"
    assert replies[0]["type"] == "oneshot"
    mock_context.job_queue.run_once.assert_called_once()

    # 2. LLM outputs cancellation
    sched_cancel_spec = {
        "action": "cancel",
        "schedule_id": created_id,
    }
    cancel_eval_res = ("Törölve az emlékeztető!", None, None, sched_cancel_spec)
    with patch.object(llm, "evaluate_and_reply", AsyncMock(return_value=cancel_eval_res)):
        await handlers._execute_evaluation(mock_context, p.group_chat_id, is_direct_trigger=True, trigger_msg_id=1235)

    assert len(s.get_scheduled_replies()) == 0


@pytest.mark.asyncio
async def test_scheduled_reply_callback_oneshot_and_periodic(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_bot = MagicMock()
    mock_bot.id = 9999
    mock_bot.first_name = "Pletykas"
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=5001))
    mock_context = MagicMock(bot=mock_bot, job_queue=MagicMock())

    # 1. Oneshot reply execution
    s_id = s.add_scheduled_reply({
        "id": "sched_oneshot_1",
        "type": "oneshot",
        "chat_id": p.group_chat_id,
        "target_msg_id": 42,
        "target_time": "2026-09-26T14:00:00+00:00",
        "interval_str": None,
        "interval_spec": None,
        "description": "Remind Bob to submit timesheet",
        "created_at": "2026-09-26T12:00:00+00:00",
    })

    job_mock = MagicMock()
    job_mock.data = {"schedule_id": s_id}
    mock_context.job = job_mock

    with patch.object(llm, "generate_scheduled_reply", AsyncMock(return_value="Hahó Bob, ne feledd a timesheetet! 😉")):
        await handlers._scheduled_reply_callback(mock_context)

    mock_bot.send_message.assert_awaited()
    assert mock_bot.send_message.call_args.kwargs["chat_id"] == p.group_chat_id
    assert mock_bot.send_message.call_args.kwargs["reply_to_message_id"] == 42
    assert "timesheetet" in mock_bot.send_message.call_args.kwargs["text"]
    # Oneshot reply is removed from state after firing
    assert s.get_scheduled_reply(s_id) is None

    # 2. Periodic reply execution
    p_id = s.add_scheduled_reply({
        "id": "sched_periodic_1",
        "type": "periodic",
        "chat_id": p.group_chat_id,
        "target_msg_id": None,
        "target_time": "2026-09-26T14:00:00+00:00",
        "interval_str": "1 day",
        "interval_spec": {"years": 0, "months": 0, "days": 1, "hours": 0, "minutes": 0, "seconds": 0},
        "description": "Daily standup reminder",
        "created_at": "2026-09-26T12:00:00+00:00",
    })
    job_mock.data = {"schedule_id": p_id}

    with patch.object(llm, "generate_scheduled_reply", AsyncMock(return_value="Kezdődik a napi standup! 🚀")):
        await handlers._scheduled_reply_callback(mock_context)

    # Periodic reply is NOT removed; its target_time is updated to the future
    updated_p = s.get_scheduled_reply(p_id)
    assert updated_p is not None
    assert datetime.fromisoformat(updated_p["target_time"]) > s.get_current_time()


@pytest.mark.asyncio
async def test_scheduled_reply_callback_sleeping(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_bot = MagicMock()
    mock_bot.id = 9999
    mock_bot.first_name = "Pletykas"
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=6001))
    mock_context = MagicMock(bot=mock_bot, job_queue=MagicMock())

    # Enable sleep mode
    s.set_sleep_schedule(enabled=True, sleep_start="00:00", sleep_end="23:59")
    assert s.is_sleeping() is True

    # Periodic reply during sleep hours: advances target_time, does NOT send message
    p_id = s.add_scheduled_reply({
        "id": "sched_periodic_quiet",
        "type": "periodic",
        "chat_id": p.group_chat_id,
        "target_msg_id": None,
        "target_time": "2026-09-26T14:00:00+00:00",
        "interval_str": "1 day",
        "interval_spec": {"years": 0, "months": 0, "days": 1, "hours": 0, "minutes": 0, "seconds": 0},
        "description": "Periodic check",
        "created_at": "2026-09-26T12:00:00+00:00",
    })
    job_mock = MagicMock()
    job_mock.data = {"schedule_id": p_id}
    mock_context.job = job_mock

    await handlers._scheduled_reply_callback(mock_context)
    mock_bot.send_message.assert_not_awaited()
    assert s.get_scheduled_reply(p_id) is not None

    # Oneshot reply during sleep hours: executes anyway (explicit user request)
    o_id = s.add_scheduled_reply({
        "id": "sched_oneshot_quiet",
        "type": "oneshot",
        "chat_id": p.group_chat_id,
        "target_msg_id": 99,
        "target_time": "2026-09-26T14:00:00+00:00",
        "interval_str": None,
        "interval_spec": None,
        "description": "Wake up Alice",
        "created_at": "2026-09-26T12:00:00+00:00",
    })
    job_mock.data = {"schedule_id": o_id}
    with patch.object(llm, "generate_scheduled_reply", AsyncMock(return_value="Ébresztő Alice!")):
        await handlers._scheduled_reply_callback(mock_context)
    mock_bot.send_message.assert_awaited_once()


def test_load_and_schedule_pending_replies(test_setup):
    from datetime import timedelta
    p, s, m, llm, handlers = test_setup
    mock_jq = MagicMock()
    mock_context = MagicMock(job_queue=mock_jq)

    now = s.get_current_time()
    tz = s.get_tzinfo()

    # 1. Future oneshot (> now) -> scheduled
    future_dt = now + timedelta(hours=2)
    s.add_scheduled_reply({
        "id": "sched_future",
        "type": "oneshot",
        "chat_id": p.group_chat_id,
        "target_time": future_dt.isoformat(),
        "description": "Future job",
    })

    # 2. Overdue oneshot (< 15 min ago) -> scheduled to fire immediately (delay 1.0s)
    recent_past = now - timedelta(minutes=5)
    s.add_scheduled_reply({
        "id": "sched_recent_past",
        "type": "oneshot",
        "chat_id": p.group_chat_id,
        "target_time": recent_past.isoformat(),
        "description": "Recent past job",
    })

    # 3. Expired oneshot (>= 15 min ago) -> discarded
    old_past = now - timedelta(hours=2)
    s.add_scheduled_reply({
        "id": "sched_expired",
        "type": "oneshot",
        "chat_id": p.group_chat_id,
        "target_time": old_past.isoformat(),
        "description": "Expired job",
    })

    # 4. Periodic in past -> advanced to future and scheduled
    s.add_scheduled_reply({
        "id": "sched_periodic_past",
        "type": "periodic",
        "chat_id": p.group_chat_id,
        "target_time": old_past.isoformat(),
        "interval_str": "1 day",
        "interval_spec": {"years": 0, "months": 0, "days": 1, "hours": 0, "minutes": 0, "seconds": 0},
        "description": "Periodic past job",
    })

    handlers.load_and_schedule_pending_replies(mock_context)

    # Expired job should be removed
    assert s.get_scheduled_reply("sched_expired") is None
    # Other 3 should be active
    assert s.get_scheduled_reply("sched_future") is not None
    assert s.get_scheduled_reply("sched_recent_past") is not None
    assert s.get_scheduled_reply("sched_periodic_past") is not None
    assert mock_jq.run_once.call_count == 3


@pytest.mark.asyncio
async def test_cmd_scheduled_and_status(test_setup):
    p, s, m, llm, handlers = test_setup
    mock_msg = MagicMock()
    mock_msg.reply_text = AsyncMock()
    mock_update = MagicMock()
    mock_update.effective_user = MagicMock(id=p.admin_user_ids[0])
    mock_update.effective_chat = MagicMock(type="private")
    mock_update.effective_message = mock_msg
    mock_context = MagicMock(args=[], job_queue=MagicMock())

    # 1. /status with 0 scheduled replies
    await handlers.cmd_status(mock_update, mock_context)
    status_text = mock_msg.reply_text.call_args[0][0]
    assert "Scheduled Replies: <code>0 active (Next: None)</code>" in status_text

    # 2. /scheduled when empty
    mock_msg.reply_text.reset_mock()
    await handlers.cmd_scheduled(mock_update, mock_context)
    assert "No active scheduled replies" in mock_msg.reply_text.call_args[0][0]

    # 3. Add scheduled reply and list
    s_id = s.add_scheduled_reply({
        "id": "sched_test_101",
        "type": "oneshot",
        "chat_id": p.group_chat_id,
        "target_time": "2026-10-01T10:00:00+00:00",
        "description": "Send monthly report",
    })

    mock_msg.reply_text.reset_mock()
    await handlers.cmd_scheduled(mock_update, mock_context)
    list_text = mock_msg.reply_text.call_args[0][0]
    assert "sched_test_101" in list_text
    assert "Send monthly report" in list_text

    # 4. Status now reflects 1 active
    mock_msg.reply_text.reset_mock()
    await handlers.cmd_status(mock_update, mock_context)
    status_text2 = mock_msg.reply_text.call_args[0][0]
    assert "Scheduled Replies: <code>1 active" in status_text2

    # 5. Cancel scheduled reply
    mock_msg.reply_text.reset_mock()
    mock_context.args = ["cancel", s_id]
    await handlers.cmd_scheduled(mock_update, mock_context)
    cancel_text = mock_msg.reply_text.call_args[0][0]
    assert "has been cancelled" in cancel_text
    assert s.get_scheduled_reply(s_id) is None

    # 6. Cancel nonexistent schedule
    mock_msg.reply_text.reset_mock()
    mock_context.args = ["cancel", "nonexistent_id"]
    await handlers.cmd_scheduled(mock_update, mock_context)
    assert "not found" in mock_msg.reply_text.call_args[0][0]

