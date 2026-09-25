import asyncio
import base64
import html
import io
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import uuid

from telegram import ReactionTypeEmoji, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from llm import LLMClient, compress_image
from memory import MemoryManager, validate_memory_dict
from params import Params
from state import CHAT_HISTORY_SIZE, StateManager

logger = logging.getLogger(__name__)

HELP_MESSAGE = """🤖 <b>Pletykas Admin Commands</b>

<b>Status & Stats</b>
/status - Current status, active parameters, observed count, reply count
/debug [on|off] - Toggle debug logging of raw LLM request and response bodies

<b>Behavior & Timing</b>
/talkativeness [1-10] - View or adjust autonomous conversation entry level
/cooldown [sec] - View or adjust message debounce timer (0 disables)
/nicknames [nick1, nick2...] - View or configure comma-separated nicknames that trigger the bot
/language [lang] - View or change bot communication language (e.g. English, Hungarian)
/timezone [tz] - View or adjust timezone (e.g. Europe/Budapest)
/sleep [on|off] [start] [end] - View or configure night quiet hours (e.g. /sleep on 23:00 07:00)
/spontaneous [on|off] - View or toggle periodic spontaneous messages
/spontaneous_interval [min] [max] - Adjust random timer interval in hours (e.g. /spontaneous_interval 2 4)
/spontaneous_now - Instantly trigger a spontaneous message or poll to the group

<b>Prompt, Grounding & Vision</b>
/prompt - Upload system prompt text file as-is
/prompt load - Expect a system prompt text file upload to validate and load
/prompt reset - Reset to default persona prompt
/grounding [on|off] - Toggle Google Search grounding for real-time web info
/image_large [on|off] - Toggle instant large model for image interpretation (ON: large instant, OFF: small model first)

<b>Memory Management</b>
/memories - Upload memories JSON file as-is
/memories load - Expect a memories JSON file upload to validate and load
/cancel - Abort a pending memory or prompt upload
/curate - Trigger immediate LLM reflection and consolidation of recent chat
/help - Show this guide"""

TELEGRAM_ALLOWED_REACTION_EMOJIS = {
    "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "🤬", "😢", "🎉", "🤩", "🤮", "💩", "🙏", "👌", "🕊",
    "🤡", "🥱", "🥴", "😍", "🐳", "❤‍🔥", "🌚", "🌭", "💯", "🤣", "⚡", "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋",
    "🖕", "😈", "😴", "😭", "🤓", "👻", "👨‍💻", "👀", "🎃", "🙈", "😇", "😨", "🤝", "✍", "🤗", "🫡", "🎅", "🎄", "☃",
    "💅", "🤪", "🗿", "🆒", "💘", "🙉", "🦄", "😘", "💊", "🙊", "😎", "👾", "🤷‍♂", "🤷", "🤷‍♀", "😡",
}

EMOJI_REACTION_MAP = {
    "😄": "😁", "😃": "😁", "😀": "😁", "😊": "😁", "🙂": "😁", "☺": "😁",
    "😂": "🤣", "😆": "🤣", "😹": "🤣",
    "❤️": "❤", "💕": "❤", "💖": "❤", "💗": "❤", "💓": "❤", "💞": "❤",
    "🥺": "😢", "😿": "😢", "😥": "😢", "😓": "😢", "🙁": "😢", "☹": "😢",
    "💀": "👻", "☠": "👻",
    "✨": "⚡", "⭐": "⚡", "🌟": "⚡",
    "🍻": "🍾", "🍺": "🍾", "🥂": "🍾",
    "😠": "😡", "👿": "😈",
    "🤐": "😐", "😶": "😐", "😑": "😐",
    "😮": "😱", "😯": "😱", "😲": "😱",
    "😏": "😈", "😉": "😘", "😜": "🤪", "😝": "🤪", "😋": "😍", "🤤": "😍",
    "🤭": "🙈", "🙄": "🤨", "😒": "🤨", "😬": "🥴", "🫠": "🥴", "🥳": "🎉",
    "🧐": "🤔", "💪": "🔥",
}
def split_into_html_pre_chunks(raw_text: str, header: str = "", max_escaped_len: int = 3500) -> List[str]:
    """Splits raw text into chunks safely wrapped in <pre> tags without exceeding Telegram's 4096-char limit."""
    lines = raw_text.splitlines(keepends=True)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for line in lines:
        while line:
            esc_line = html.escape(line)
            if len(esc_line) <= max_escaped_len:
                if current_len + len(esc_line) > max_escaped_len and current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = [line]
                    current_len = len(esc_line)
                else:
                    current_chunk.append(line)
                    current_len += len(esc_line)
                break
            else:
                cut = max_escaped_len // 2
                while cut > 0 and len(html.escape(line[:cut])) > max_escaped_len:
                    cut -= 10
                if cut <= 0:
                    cut = 1
                segment = line[:cut]
                line = line[cut:]
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.append(segment)

    if current_chunk:
        chunks.append("".join(current_chunk))

    if not chunks:
        chunks = [""]

    total = len(chunks)
    messages: List[str] = []
    for i, chunk in enumerate(chunks):
        esc_chunk = html.escape(chunk)
        if total == 1:
            prefix = f"{header}\n" if header else ""
        else:
            prefix = f"{header} (Part {i + 1}/{total}):\n" if header else f"<b>(Part {i + 1}/{total}):</b>\n"
        messages.append(f"{prefix}<pre>{esc_chunk}</pre>")
    return messages



class BotHandlers:
    def __init__(self, params: Params, state: StateManager, memory: MemoryManager, llm: LLMClient):
        self.params = params
        self.state = state
        self.memory = memory
        self.llm = llm
        self._debounce_jobs: Dict[int, Any] = {}
        self._active_evaluations: set[int] = set()
        self._waiting_for_memory_upload: set[int] = set()
        self._waiting_for_prompt_upload: set[int] = set()
    def _log_debug_group_msg(self, direction: str, text: str) -> None:
        if self.state.is_debug_mode():
            banner = f"TELEGRAM GROUP {direction.upper()}"
            sys.stdout.write(f"\n--- [DEBUG {banner}] ---\n{text}\n--- [END DEBUG {banner}] ---\n")
            sys.stdout.flush()
            logger.debug("%s:\n%s", banner, text)

    @staticmethod
    def resolve_reaction_emoji(raw_emoji: str) -> Optional[str]:
        e = raw_emoji.strip()
        if e in EMOJI_REACTION_MAP:
            e = EMOJI_REACTION_MAP[e]
        norm = e.replace("\ufe0f", "")
        if norm in TELEGRAM_ALLOWED_REACTION_EMOJIS:
            return norm
        if e in TELEGRAM_ALLOWED_REACTION_EMOJIS:
            return e
        if norm in EMOJI_REACTION_MAP:
            mapped = EMOJI_REACTION_MAP[norm]
            return mapped.replace("\ufe0f", "")
        return None


    # --- Authorization & Routing Middleware ---

    def _is_admin(self, user_id: Optional[int]) -> bool:
        return user_id is not None and self.params.is_admin(user_id)

    async def _check_group_authorization(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Verifies if group is authorized. Leaves unauthorized groups immediately."""
        chat = update.effective_chat
        if not chat:
            return False

        if chat.type in ("group", "supergroup"):
            # Check supergroup migration
            msg = update.effective_message
            migrate_id = getattr(msg, "migrate_to_chat_id", None) if msg else None
            if isinstance(migrate_id, int) and migrate_id != 0:
                new_id = migrate_id
                logger.info("Migrated group from %d to supergroup %d", chat.id, new_id)
                self.params.group_chat_id = new_id
                self.state.set_group_chat_id(new_id)
                return True

            if not self.params.is_group_authorized(chat.id):
                logger.warning("Bot added to unauthorized chat %d (%s). Leaving...", chat.id, chat.title)
                try:
                    await context.bot.leave_chat(chat.id)
                except Exception as e:
                    logger.error("Failed to leave unauthorized chat %d: %s", chat.id, e)
                return False

            # Auto-normalize if chat ID has -100 prefix but was configured without it
            if self.params.group_chat_id != chat.id:
                logger.info("Normalizing group_chat_id from %d to full Telegram supergroup ID %d", self.params.group_chat_id, chat.id)
                self.params.group_chat_id = chat.id
                self.state.set_group_chat_id(chat.id)
        return True

    # --- Spontaneous Messages & Inactivity Timer ---

    def schedule_spontaneous_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Schedules spontaneous revival message at a random interval between min_hours and max_hours."""
        if not self.state.is_spontaneous_enabled():
            if context.job_queue:
                for job in context.job_queue.get_jobs_by_name("spontaneous_revival"):
                    job.schedule_removal()
            return

        if not context.job_queue:
            logger.debug("JobQueue not initialized, cannot schedule spontaneous job")
            return

        # Cancel any existing spontaneous jobs
        for job in context.job_queue.get_jobs_by_name("spontaneous_revival"):
            job.schedule_removal()

        spont = self.state.get_spontaneous_settings()
        min_hours = float(spont.get("min_hours", 2.0))
        max_hours = float(spont.get("max_hours", 4.0))
        if min_hours <= 0:
            min_hours = 2.0
        if max_hours < min_hours:
            max_hours = min_hours

        delay_sec = random.uniform(min_hours * 3600.0, max_hours * 3600.0)

        context.job_queue.run_once(
            self._spontaneous_callback,
            when=delay_sec,
            name="spontaneous_revival",
            data={"group_chat_id": self.params.group_chat_id},
        )
        logger.info("Scheduled spontaneous revival in %.2f hours (%.0f seconds)", delay_sec / 3600.0, delay_sec)
    async def _spontaneous_callback(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Triggered periodically at random intervals to post spontaneous messages."""
        if self.state.is_sleeping():
            logger.info("Spontaneous trigger fired during sleep hours; skipping and rescheduling.")
            self.schedule_spontaneous_job(context)
            return

        await self.trigger_spontaneous_message(context)
        self.schedule_spontaneous_job(context)

    async def trigger_spontaneous_message(self, context: ContextTypes.DEFAULT_TYPE) -> Tuple[bool, str]:
        """Generates and dispatches a spontaneous conversation starter (text or poll) to the group."""
        group_id = self.params.group_chat_id
        if not group_id:
            logger.warning("Group chat ID not configured, cannot send spontaneous message")
            return False, "Group chat ID is not configured."

        logger.info("Executing spontaneous conversation revival for group %d", group_id)
        transcript = self._format_transcript(self.state.get_chat_history())
        memory_ctx = self.memory.format_for_context()
        sys_prompt = self.state.get_effective_system_prompt()
        tz_str = self.state.get_timezone()

        try:
            response = await self.llm.generate_spontaneous_message(
                system_prompt=sys_prompt,
                memory_context=memory_ctx,
                transcript=transcript,
                timezone_str=tz_str,
            )
        except Exception as e:
            logger.error("LLM error generating spontaneous message: %s", e)
            return False, f"LLM error: {e}"

        if not response:
            logger.warning("LLM returned empty spontaneous message")
            return False, "LLM returned empty spontaneous message."

        # Check for poll format
        poll_match = re.search(r"<POLL>([\s\S]*?)</POLL>", response)
        if poll_match:
            block = poll_match.group(1).strip()
            q = ""
            opts = []
            for line in block.splitlines():
                line = line.strip()
                if line.lower().startswith("question:"):
                    q = line[9:].strip()
                elif line.startswith("-") or line.startswith("•") or line.startswith("*"):
                    opt = line.lstrip("-•* ").strip()
                    if opt:
                        opts.append(opt)

            if q and len(opts) >= 2:
                try:
                    sent = await context.bot.send_poll(
                        chat_id=group_id,
                        question=q,
                        options=opts[:10],
                        is_anonymous=False,
                    )
                    now_str = self.state.get_current_time_str()
                    if self.state.is_debug_mode():
                        self._log_debug_group_msg(
                            "OUTGOING",
                            f"Chat ID: {group_id} | Message ID: {sent.message_id}\nType: Spontaneous Poll\nQuestion: {q}\nOptions: {', '.join(opts[:10])}",
                        )
                    poll_text = f"[Poll: {q}] Options: {', '.join(opts[:10])}"
                    bot_name = context.bot.first_name if isinstance(getattr(context.bot, "first_name", None), str) else "Pletykas"
                    bot_id = context.bot.id if isinstance(getattr(context.bot, "id", None), int) else 0
                    entry = {
                        "id": sent.message_id,
                        "from_user_id": bot_id,
                        "from_user_name": bot_name,
                        "timestamp_epoch": time.time(),
                        "timestamp_str": now_str,
                        "reply_to_msg_id": None,
                        "reply_to_user_name": None,
                        "text": poll_text,
                        "media_type": "poll",
                        "media_b64": None,
                    }
                    self.state.append_chat_message(entry)
                    self.state.append_memory_message(entry)
                    return True, poll_text
                except Exception as e:
                    logger.error("Failed to send spontaneous poll: %s", e)
                    return False, f"Failed to send poll: {e}"

        # Standard text message
        try:
            try:
                sent = await context.bot.send_message(chat_id=group_id, text=response, parse_mode=ParseMode.HTML)
            except Exception as pe:
                logger.debug("Failed sending spontaneous message with HTML parse mode (%s), falling back to plain text", pe)
                sent = await context.bot.send_message(chat_id=group_id, text=response)
            now_str = self.state.get_current_time_str()
            bot_name = context.bot.first_name if isinstance(getattr(context.bot, "first_name", None), str) else "Pletykas"
            if self.state.is_debug_mode():
                self._log_debug_group_msg(
                    "OUTGOING",
                    f"Chat ID: {group_id} | Message ID: {sent.message_id}\nType: Spontaneous Text Message\nText: {response}",
                )
            bot_id = context.bot.id if isinstance(getattr(context.bot, "id", None), int) else 0
            entry = {
                "id": sent.message_id,
                "from_user_id": bot_id,
                "from_user_name": bot_name,
                "timestamp_epoch": time.time(),
                "timestamp_str": now_str,
                "reply_to_msg_id": None,
                "reply_to_user_name": None,
                "text": response,
                "media_type": "none",
                "media_b64": None,
            }
            self.state.append_chat_message(entry)
            self.state.append_memory_message(entry)
            return True, response
        except Exception as e:
            logger.error("Failed to send spontaneous message: %s", e)
            return False, f"Failed to send message: {e}"

    # --- Memory Curation ---

    async def trigger_curation(self) -> Dict[str, Any]:
        """Runs reflection and curation of recent chat messages into permanent memory."""
        mem_history = self.state.get_memory_history()
        transcript = self._format_transcript(mem_history)
        current_memories = self.memory.get_all_memories()

        curation = await self.llm.curate_memory(current_memories, transcript)

        # Create timestamped backup before committing updates
        self.memory.create_backup()

        added_facts = 0
        updated_facts = 0
        discarded_facts = 0
        added_dyn = 0
        discarded_dyn = 0
        added_jokes = 0
        discarded_jokes = 0

        # Facts
        for f in curation.get("facts_to_add", []):
            if f.get("topic") and f.get("content"):
                self.memory.add_memory(f["topic"], f["content"])
                added_facts += 1

        for f in curation.get("facts_to_update", []):
            if f.get("topic") and f.get("content"):
                if self.memory.update_memory(f["topic"], f["content"]):
                    updated_facts += 1
                else:
                    self.memory.add_memory(f["topic"], f["content"])
                    added_facts += 1

        for target in curation.get("facts_to_discard", []):
            if self.memory.discard_memory(str(target)):
                discarded_facts += 1

        # Dynamics
        for d in curation.get("dynamics_to_add", []):
            if d.get("members") and d.get("relation"):
                self.memory.add_dynamic(d["members"], d["relation"])
                added_dyn += 1

        for target in curation.get("dynamics_to_discard", []):
            if self.memory.discard_dynamic(str(target)):
                discarded_dyn += 1

        # Jokes
        for j in curation.get("jokes_to_add", []):
            if j.get("title") and j.get("context"):
                self.memory.add_inside_joke(j["title"], j["context"])
                added_jokes += 1

        for target in curation.get("jokes_to_discard", []):
            if self.memory.discard_inside_joke(str(target)):
                discarded_jokes += 1
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.memory.set_last_curated_at(now_iso)

        summary = {
            "added_facts": added_facts,
            "updated_facts": updated_facts,
            "discarded_facts": discarded_facts,
            "added_dynamics": added_dyn,
            "discarded_dynamics": discarded_dyn,
            "added_jokes": added_jokes,
            "discarded_jokes": discarded_jokes,
        }
        logger.info("Memory consolidation completed: %s", summary)
        return summary

    # --- Message Helpers & Formatting ---

    def _format_transcript(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return "(No recent messages)"
        lines = []
        for msg in history:
            reply_part = f" (replying to {msg['reply_to_user_name']})" if msg.get("reply_to_user_name") else ""
            lines.append(f"[ID: {msg.get('id')}] [{msg.get('timestamp_str')}] [{msg.get('from_user_name')}]{reply_part}: {msg.get('text')}")
        return "\n".join(lines)

    def _is_direct_trigger(
        self,
        message: Any,
        bot_username: str = "",
        bot_name: str = "",
        bot_id: Optional[int] = None,
    ) -> bool:
        """Determines if a message is an explicit direct trigger for the bot."""
        # 1. Replying directly to bot's message
        if message.reply_to_message and message.reply_to_message.from_user:
            from_user = message.reply_to_message.from_user
            if from_user.is_bot:
                if bot_id and from_user.id == bot_id:
                    return True
                if bot_username and from_user.username and from_user.username.lower() == bot_username.lower():
                    return True

        # 2. Text / caption analysis
        text = message.text or message.caption or ""

        # 3. Message entities (Telegram mentions)
        if message.entities and text:
            for entity in message.entities:
                if entity.type == "mention" and bot_username:
                    mention_text = text[entity.offset: entity.offset + entity.length].lstrip("@").lower()
                    if mention_text == bot_username.lower():
                        return True
                elif entity.type == "text_mention" and bot_id:
                    if entity.user and entity.user.id == bot_id:
                        return True

        if not text:
            return False

        # 4. Candidate names: telegram username (with or without @), bot display name, persona name, and configured nicknames
        candidates: List[str] = []
        if bot_username and isinstance(bot_username, str):
            candidates.append(bot_username)
        if bot_name and isinstance(bot_name, str):
            candidates.append(bot_name)
        for default_name in ("Pletykás", "Pletykas"):
            if default_name not in candidates:
                candidates.append(default_name)
        candidates.extend(self.state.get_nicknames())

        # Match each candidate with word / punctuation boundaries
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            clean = candidate.strip().lstrip("@").strip()
            if not clean:
                continue
            pattern = rf"(?:^|[^\w@])@?{re.escape(clean)}(?:[^\w]|$)"
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False
    # --- Debounce & Evaluation Pipeline ---

    async def _debounce_callback(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Called when cooldown debounce timer finishes quietly."""
        job = context.job
        data = job.data or {}
        chat_id = data.get("chat_id")
        trigger_msg_id = data.get("trigger_msg_id")
        self._debounce_jobs.pop(chat_id, None)

        await self._execute_evaluation(
            context=context,
            chat_id=chat_id,
            is_direct_trigger=False,
            trigger_msg_id=trigger_msg_id,
        )

    async def _execute_evaluation(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        is_direct_trigger: bool,
        trigger_msg_id: int,
    ) -> None:
        """Performs LLM conversation evaluation and dispatches replies, reactions, or images."""
        if chat_id in self._active_evaluations:
            logger.debug("Evaluation already in progress for chat %d, skipping duplicate", chat_id)
            return

        # Check sleep schedule for passive conversation entry
        if not is_direct_trigger and self.state.is_sleeping():
            logger.debug("Bot is sleeping; skipping autonomous conversation entry")
            return

        self._active_evaluations.add(chat_id)
        bot_username = context.bot.username or ""

        try:
            transcript = self._format_transcript(self.state.get_chat_history())
            memory_ctx = self.memory.format_for_context()
            sys_prompt = self.state.get_effective_system_prompt()
            talkativeness = self.state.get_talkativeness()

            # Find the trigger message in history to check for photo media
            trigger_entry = None
            for item in reversed(self.state.get_chat_history()):
                if item.get("id") == trigger_msg_id:
                    trigger_entry = item
                    break

            # Check if this message or its context involves an image
            photo_entry = None
            # 1. The trigger message itself is a photo
            if trigger_entry and trigger_entry.get("media_type") == "photo" and trigger_entry.get("media_b64"):
                photo_entry = trigger_entry
            # 2. The trigger message is a reply to an earlier photo
            elif trigger_entry and trigger_entry.get("reply_to_msg_id"):
                reply_target_id = trigger_entry.get("reply_to_msg_id")
                for item in reversed(self.state.get_chat_history()):
                    if item.get("id") == reply_target_id and item.get("media_type") == "photo" and item.get("media_b64"):
                        photo_entry = item
                        break
            # 3. The trigger message explicitly references an image and there is an undescribed photo in recent history
            if not photo_entry and trigger_entry:
                txt = (trigger_entry.get("text") or "").lower()
                image_keywords = ["kép", "kep", "fotó", "foto", "photo", "image", "picture", "rajz", "ábra", "abra"]
                if any(kw in txt for kw in image_keywords):
                    for item in reversed(self.state.get_chat_history()):
                        if item.get("media_type") == "photo" and item.get("media_b64"):
                            photo_entry = item
                            break

            reply_text: Optional[str] = None
            reaction: Optional[Tuple[str, Optional[int]]] = None
            image_spec: Optional[Dict[str, str]] = None

            if photo_entry:
                # Multimodal evaluation via Vision Model
                photo_bytes = base64.b64decode(photo_entry["media_b64"])
                user_caption = trigger_entry.get("text", "").replace("[Photo]", "").strip() if trigger_entry else ""
                reply_text, reaction, image_spec, img_desc = await self.llm.describe_and_reply_image(
                    system_prompt=sys_prompt,
                    memory_context=memory_ctx,
                    transcript=transcript,
                    bot_username=bot_username,
                    is_direct_trigger=is_direct_trigger,
                    talkativeness=talkativeness,
                    image_bytes=photo_bytes,
                    caption=user_caption,
                )
                # Enrich photo entry with visual description if not already enriched
                if img_desc and ("[Photo: " not in photo_entry.get("text", "")):
                    clean_caption = photo_entry.get("text", "").replace("[Photo]", "").strip()
                    photo_entry["text"] = f"[Photo: {img_desc}] {clean_caption}".strip()
                    self.state.save()
            else:
                # Standard conversational evaluation
                reply_text, reaction, image_spec = await self.llm.evaluate_and_reply(
                    system_prompt=sys_prompt,
                    memory_context=memory_ctx,
                    transcript=transcript,
                    bot_username=bot_username,
                    is_direct_trigger=is_direct_trigger,
                    talkativeness=talkativeness,
                )

            # 1. Emoji Reaction Action
            if reaction:
                raw_emoji, target_id = reaction
                target = target_id or trigger_msg_id
                resolved_emoji = self.resolve_reaction_emoji(raw_emoji)
                if not resolved_emoji:
                    logger.debug("Skipping unsupported reaction emoji: %s", raw_emoji)
                else:
                    try:
                        await context.bot.set_message_reaction(
                            chat_id=chat_id,
                            message_id=target,
                            reaction=[ReactionTypeEmoji(emoji=resolved_emoji)],
                        )
                        if self.state.is_debug_mode():
                            self._log_debug_group_msg(
                                "OUTGOING",
                                f"Chat ID: {chat_id} | Target Message ID: {target}\nType: Emoji Reaction\nReaction: {resolved_emoji}",
                            )
                        logger.info("Dispatched reaction %s on message %d", resolved_emoji, target)
                    except Exception as e:
                        err_str = str(e).lower()
                        if "reaction_invalid" in err_str or "reactions_disabled" in err_str or "reaction invalid" in err_str:
                            logger.info("Chat does not accept reaction %s on message %d: %s", resolved_emoji, target, e)
                        else:
                            logger.warning("Failed to set reaction %s on message %d: %s", resolved_emoji, target, e)
            # 2. Image Generation & Modification Action
            if image_spec:
                # Suppress typing and send upload_photo action strictly while generating
                try:
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
                except Exception:
                    pass

                prompt = image_spec.get("prompt", "")
                caption = image_spec.get("caption") or reply_text or ""
                source = image_spec.get("source", "new")
                mode = image_spec.get("mode", "generate")

                base_image_bytes = None
                if mode == "modify" or source == "reply":
                    # If a photo_entry was already detected in current context, prioritize it
                    if photo_entry and photo_entry.get("media_b64"):
                        base_image_bytes = base64.b64decode(photo_entry["media_b64"])
                    else:
                        # Search for source image in recent chat history
                        for item in reversed(self.state.get_chat_history()):
                            if (source == "reply" and item.get("media_type") == "photo") or (
                                str(item.get("id")) == str(source) and item.get("media_type") == "photo"
                            ):
                                if item.get("media_b64"):
                                    base_image_bytes = base64.b64decode(item["media_b64"])
                                    break
                try:
                    logger.info(
                        "Generating/modifying image with model '%s' (mode=%s, source=%s, prompt=%s)",
                        self.params.model_image_name,
                        mode,
                        source,
                        prompt[:60],
                    )
                    gen_bytes = await self.llm.generate_image(prompt=prompt, base_image_bytes=base_image_bytes)
                    compressed_gen = compress_image(gen_bytes, max_dim=1280, quality=85)
                    gen_b64 = base64.b64encode(compressed_gen).decode("utf-8")

                    try:
                        sent_photo = await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=gen_bytes,
                            caption=caption,
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        sent_photo = await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=gen_bytes,
                            caption=caption,
                        )
                    now_str = self.state.get_current_time_str()

                    if self.state.is_debug_mode():
                        reply_info = f" | Reply to msg ID: {trigger_msg_id}" if is_direct_trigger else ""
                        self._log_debug_group_msg(
                            "OUTGOING",
                            f"Chat ID: {chat_id} | Message ID: {sent_photo.message_id}{reply_info}\nType: Generated Photo\nCaption: {caption}\nPrompt: {prompt}",
                        )
                    # Record generated image in history
                    photo_entry = {
                        "id": sent_photo.message_id,
                        "from_user_id": context.bot.id,
                        "from_user_name": context.bot.first_name or "Pletykas",
                        "timestamp_epoch": time.time(),
                        "timestamp_str": now_str,
                        "reply_to_msg_id": trigger_msg_id if is_direct_trigger else None,
                        "reply_to_user_name": None,
                        "text": f"[Generated Image: {prompt}] {caption}".strip(),
                        "media_type": "photo",
                        "media_b64": gen_b64,
                    }
                    self.state.append_chat_message(photo_entry)
                    self.state.append_memory_message(photo_entry)

                    # Caption already accompanied photo; avoid duplicate text message
                    reply_text = None
                except Exception as e:
                    logger.error("Failed to generate or send image: %s", e)
                    if is_direct_trigger:
                        reply_text = "sorry, I couldn't generate the image... 🎨❌"

            # 3. Text Message Action
            if reply_text:
                reply_to_id = trigger_msg_id if is_direct_trigger else None
                try:
                    sent_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        text=reply_text,
                        reply_to_message_id=reply_to_id,
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as pe:
                    logger.debug("Failed sending reply with HTML parse mode (%s), falling back to plain text", pe)
                    sent_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        text=reply_text,
                        reply_to_message_id=reply_to_id,
                    )
                now_str = self.state.get_current_time_str()
                if self.state.is_debug_mode():
                    reply_info = f" | Reply to msg ID: {reply_to_id}" if reply_to_id else ""
                    self._log_debug_group_msg(
                        "OUTGOING",
                        f"Chat ID: {chat_id} | Message ID: {sent_msg.message_id}{reply_info}\nType: Text Message\nText: {reply_text}",
                    )

                entry = {
                    "id": sent_msg.message_id,
                    "from_user_id": context.bot.id,
                    "from_user_name": context.bot.first_name or "Pletykas",
                    "timestamp_epoch": time.time(),
                    "timestamp_str": now_str,
                    "reply_to_msg_id": reply_to_id,
                    "reply_to_user_name": None,
                    "text": reply_text,
                    "media_type": "none",
                    "media_b64": None,
                }
                self.state.append_chat_message(entry)
                self.state.append_memory_message(entry)

        except Exception as e:
            logger.error("Error during evaluation execution: %s", e, exc_info=True)
            if is_direct_trigger:
                try:
                    sent_err = await context.bot.send_message(
                        chat_id=chat_id,
                        text="sorry, got a bit tangled up in my thoughts... I'll be right back! 💫",
                        reply_to_message_id=trigger_msg_id,
                    )
                    if self.state.is_debug_mode():
                        self._log_debug_group_msg(
                            "OUTGOING",
                            f"Chat ID: {chat_id} | Message ID: {sent_err.message_id} | Reply to msg ID: {trigger_msg_id}\nType: Error Fallback Message\nText: sorry, got a bit tangled up in my thoughts... I'll be right back! 💫",
                        )
                except Exception:
                    pass
        finally:
            self._active_evaluations.discard(chat_id)

    # --- Incoming Message Handlers ---

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles messages across group and private chats."""
        if not await self._check_group_authorization(update, context):
            return

        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return

        # Private chat handling
        if chat.type == "private":
            user_id = update.effective_user.id if update.effective_user else None
            if not self._is_admin(user_id):
                logger.warning("Unauthorized private message from user %s", user_id)
                return

            # Check if admin is currently waiting to upload a memories JSON file
            if user_id in self._waiting_for_memory_upload:
                if message.text and message.text.strip().lower() == "/cancel":
                    self._waiting_for_memory_upload.discard(user_id)
                    await message.reply_text("❌ Memory upload cancelled.")
                    return
                if message.document:
                    await self._process_memory_upload(update, context, message.document)
                    return
                await message.reply_text("⚠️ Expecting a JSON file document upload. Send /cancel to abort.")
                return

            # Check if admin is currently waiting to upload a prompt text file
            if user_id in self._waiting_for_prompt_upload:
                if message.text and message.text.strip().lower() == "/cancel":
                    self._waiting_for_prompt_upload.discard(user_id)
                    await message.reply_text("❌ Prompt upload cancelled.")
                    return
                if message.document:
                    await self._process_prompt_upload(update, context, message.document)
                    return
                await message.reply_text("⚠️ Expecting a text file document upload for system prompt. Send /cancel to abort.")
                return

            # Also support direct document upload with caption "/memories load" or "/prompt load"
            caption = (message.caption or "").strip().lower()
            if message.document and caption.startswith("/memories load"):
                await self._process_memory_upload(update, context, message.document)
                return
            if message.document and caption.startswith("/prompt load"):
                await self._process_prompt_upload(update, context, message.document)
                return
            # Private non-command messages give a helpful pointer
            await message.reply_text("👋 Hello! Use /help to see available administrative commands.")
            return
        # Group chat handling
        if chat.id != self.params.group_chat_id:
            return

        # Ignore bot's own messages
        if message.from_user and message.from_user.id == context.bot.id:
            return


        # Ingest media and text
        media_type = "none"
        media_b64: Optional[str] = None
        user_text = message.text or message.caption or ""

        if message.photo:
            media_type = "photo"
            try:
                photo = message.photo[-1]
                file_obj = await context.bot.get_file(photo.file_id)
                buf = io.BytesIO()
                await file_obj.download_to_memory(buf)
                compressed = compress_image(buf.getvalue(), max_dim=1280, quality=85)
                media_b64 = base64.b64encode(compressed).decode("utf-8")
                user_text = f"[Photo] {message.caption}".strip() if message.caption else "[Photo]"
            except Exception as e:
                logger.warning("Failed to download photo from message %d: %s", message.message_id, e)
                user_text = f"[Photo] {message.caption}".strip() if message.caption else "[Photo]"
        elif message.voice:
            dur = message.voice.duration or 0
            user_text = f"[Voice message {dur}s] {message.caption or ''}".strip()
            media_type = "placeholder"
        elif message.audio:
            fname = message.audio.file_name or "audio"
            user_text = f"[Audio: {fname}] {message.caption or ''}".strip()
            media_type = "placeholder"
        elif message.video:
            dur = message.video.duration or 0
            user_text = f"[Video {dur}s] {message.caption or ''}".strip()
            media_type = "placeholder"
        elif message.video_note:
            dur = message.video_note.duration or 0
            user_text = f"[Video note {dur}s]"
            media_type = "placeholder"
        elif message.document:
            fname = message.document.file_name or "document"
            user_text = f"[Document: {fname}] {message.caption or ''}".strip()
            media_type = "placeholder"
        elif message.sticker:
            emoji = message.sticker.emoji or ""
            user_text = f"[Sticker {emoji}]".strip()
            media_type = "placeholder"
        elif message.animation:
            user_text = f"[Animation/GIF] {message.caption or ''}".strip()
            media_type = "placeholder"

        # Record in transcript and memory history
        from_name = message.from_user.full_name if message.from_user else "User"
        reply_to_user_name = None
        if message.reply_to_message and message.reply_to_message.from_user:
            reply_to_user_name = message.reply_to_message.from_user.full_name

        date_obj = getattr(message, "date", None)
        try:
            epoch = float(date_obj.timestamp()) if (date_obj and hasattr(date_obj, "timestamp")) else time.time()
        except (TypeError, ValueError):
            epoch = time.time()

        now_str = self.state.format_time(epoch)
        msg_entry = {
            "id": int(message.message_id) if hasattr(message, "message_id") and isinstance(message.message_id, int) else 0,
            "from_user_id": int(message.from_user.id) if message.from_user and isinstance(message.from_user.id, int) else 0,
            "from_user_name": str(from_name),
            "timestamp_epoch": epoch,
            "timestamp_str": now_str,
            "reply_to_msg_id": message.reply_to_message.message_id if message.reply_to_message else None,
            "reply_to_user_name": reply_to_user_name,
            "text": user_text,
            "media_type": media_type,
            "media_b64": media_b64,
        }
        self.state.append_chat_message(msg_entry)
        self.state.append_memory_message(msg_entry)

        if self.state.is_debug_mode():
            reply_info = f" | Reply to msg ID: {message.reply_to_message.message_id}" if message.reply_to_message else ""
            media_info = f" [Media: {media_type}]" if media_type != "none" else ""
            sender_id = message.from_user.id if message.from_user else 0
            sender_name = from_name
            dbg_lines = [
                f"Chat ID: {chat.id} | Message ID: {message.message_id}{reply_info}",
                f"From: {sender_name} (ID: {sender_id})",
                f"Content{media_info}: {user_text}",
            ]
            self._log_debug_group_msg("INCOMING", "\n".join(dbg_lines))

        # Increment curation counter and trigger consolidation if threshold reached
        curation_count = self.state.increment_messages_since_last_curation()
        if curation_count >= CHAT_HISTORY_SIZE:
            self.state.set_messages_since_last_curation(0)
            asyncio.create_task(self.trigger_curation())


        # Check direct trigger
        bot_username = context.bot.username if isinstance(getattr(context.bot, "username", None), str) else ""
        bot_name = context.bot.first_name if isinstance(getattr(context.bot, "first_name", None), str) else ""
        bot_id = context.bot.id if isinstance(getattr(context.bot, "id", None), int) else None
        is_direct = self._is_direct_trigger(message, bot_username=bot_username, bot_name=bot_name, bot_id=bot_id)

        if is_direct:
            # Cancel any pending debounce timer and evaluate immediately
            if chat.id in self._debounce_jobs:
                old_job = self._debounce_jobs.pop(chat.id)
                old_job.schedule_removal()
            asyncio.create_task(
                self._execute_evaluation(
                    context=context,
                    chat_id=chat.id,
                    is_direct_trigger=True,
                    trigger_msg_id=message.message_id,
                )
            )
        else:
            # Passive observation: check sleep schedule
            if self.state.is_sleeping():
                return

            cooldown = self.state.get_cooldown_sec()
            if cooldown <= 0:
                asyncio.create_task(
                    self._execute_evaluation(
                        context=context,
                        chat_id=chat.id,
                        is_direct_trigger=False,
                        trigger_msg_id=message.message_id,
                    )
                )
            else:
                # Reset debounce timer
                if chat.id in self._debounce_jobs:
                    old_job = self._debounce_jobs.pop(chat.id)
                    old_job.schedule_removal()

                if context.job_queue:
                    job = context.job_queue.run_once(
                        self._debounce_callback,
                        when=cooldown,
                        data={"chat_id": chat.id, "trigger_msg_id": message.message_id},
                        name=f"debounce_{chat.id}",
                        job_kwargs={"id": f"debounce_{chat.id}_{uuid.uuid4().hex}"},
                    )
                    self._debounce_jobs[chat.id] = job
                else:
                    # Fallback for environments without JobQueue
                    async def _sleep_and_eval():
                        await asyncio.sleep(cooldown)
                        await self._execute_evaluation(context, chat.id, False, message.message_id)

                    asyncio.create_task(_sleep_and_eval())

    # --- Admin Slash Commands ---

    async def cmd_start_or_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return
        await update.effective_message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        sched = self.state.get_sleep_schedule()
        spont = self.state.get_spontaneous_settings()
        is_sleep = "Yes" if self.state.is_sleeping() else "No"
        sleep_status = f"{'ON' if sched.get('enabled') else 'OFF'} ({sched.get('sleep_start')} - {sched.get('sleep_end')}) (Sleeping: {is_sleep})"
        min_h = spont.get("min_hours", 2.0)
        max_h = spont.get("max_hours", 4.0)
        spont_status = f"{'ON' if spont.get('enabled') else 'OFF'} (every {min_h:g}-{max_h:g}h, random)"
        grounding_status = "ON" if self.state.is_search_grounding_active() else "OFF"
        debug_status = "ON" if self.state.is_debug_mode() else "OFF"
        nicks = self.state.get_nicknames()
        nicks_str = ", ".join(nicks) if nicks else "None"
        tl_primary = self.params.model_thinking_level or "off (default)"
        tl_large = self.params.effective_large_thinking_level or "off (default)"
        tl_image = self.params.effective_image_thinking_level or "minimal (default)"
        text = (
            f"📊 <b>Pletykas Status</b>\n"
            f"• Group Chat ID: <code>{self.params.group_chat_id}</code>\n"
            f"• Active Language: <code>{self.state.get_language()}</code>\n"
            f"• Timezone: <code>{self.state.get_timezone()}</code> ({self.state.get_current_time_str()})\n"
            f"• Talkativeness: <code>{self.state.get_talkativeness()}/10</code>\n"
            f"• Cooldown: <code>{self.state.get_cooldown_sec()}s</code>\n"
            f"• Nicknames: <code>{html.escape(nicks_str)}</code>\n"
            f"• Search Grounding: <code>{grounding_status}</code>\n"
            f"• Image Vision Model: <code>{'Large Model (Instant)' if self.state.is_image_interpretation_large_model() else 'Small Model First'}</code>\n"
            f"• Debug Mode: <code>{debug_status}</code>\n"
            f"• Sleep Schedule: <code>{sleep_status}</code>\n"
            f"• Spontaneous Messages: <code>{spont_status}</code>\n"
            f"• Models:\n"
            f"  - Primary: <code>{self.params.model_name}</code> (thinking: {tl_primary})\n"
            f"  - Vision: <code>{self.params.model_large_name}</code> (thinking: {tl_large})\n"
            f"  - Image: <code>{self.params.model_image_name}</code> (size: {self.params.model_image_size}, thinking: {tl_image})\n"
            f"• Stats:\n"
            f"  - Uncurated Messages: {self.state.get_messages_since_last_curation()}/20"
        )
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_talkativeness(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        if not args:
            cur = self.state.get_talkativeness()
            await update.effective_message.reply_text(f"🗣️ Talkativeness is currently set to: <b>{cur}/10</b>", parse_mode=ParseMode.HTML)
            return

        try:
            val = int(args[0])
            self.state.set_talkativeness(val)
            await update.effective_message.reply_text(f"✅ Talkativeness updated to: <b>{self.state.get_talkativeness()}/10</b>", parse_mode=ParseMode.HTML)
        except ValueError:
            await update.effective_message.reply_text("❌ Please specify an integer between 1 and 10.")

    async def cmd_cooldown(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        if not args:
            cur = self.state.get_cooldown_sec()
            await update.effective_message.reply_text(f"⏱️ Cooldown debounce timer is currently: <b>{cur} seconds</b>", parse_mode=ParseMode.HTML)
            return

        try:
            val = int(args[0])
            self.state.set_cooldown_sec(val)
            await update.effective_message.reply_text(f"✅ Cooldown debounce timer updated to: <b>{self.state.get_cooldown_sec()} seconds</b>", parse_mode=ParseMode.HTML)
        except ValueError:
            await update.effective_message.reply_text("❌ Please specify a non-negative integer in seconds (0 disables).")

    async def cmd_language(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        if not args:
            cur = self.state.get_language()
            await update.effective_message.reply_text(f"🌐 Active language is: <code>{html.escape(cur)}</code>", parse_mode=ParseMode.HTML)
            return

        lang = " ".join(args).strip()
        self.state.set_language(lang)
        await update.effective_message.reply_text(f"✅ Active language updated to: <code>{html.escape(lang)}</code>", parse_mode=ParseMode.HTML)


    async def cmd_nicknames(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        msg_text = update.effective_message.text or ""
        parts = msg_text.split(maxsplit=1)
        subcommand = parts[1].strip() if len(parts) > 1 else ""

        if not subcommand:
            nicks = self.state.get_nicknames()
            if nicks:
                nicks_formatted = ", ".join(nicks)
                await update.effective_message.reply_text(
                    f"🏷️ Current nicknames: <code>{html.escape(nicks_formatted)}</code>\n\n"
                    f"The bot will react directly when addressed by any of these nicknames, its name, or its Telegram username.",
                    parse_mode=ParseMode.HTML,
                )
            else:
                await update.effective_message.reply_text(
                    "🏷️ No nicknames configured. The bot currently reacts directly when addressed by its name or Telegram username.\n\n"
                    "Usage: <code>/nicknames pletyka, pleti, botika</code>\n"
                    "To clear: <code>/nicknames clear</code>",
                    parse_mode=ParseMode.HTML,
                )
            return

        if subcommand.lower() in ("clear", "reset", "none", "off"):
            self.state.set_nicknames([])
            await update.effective_message.reply_text("✅ Cleared all nicknames.", parse_mode=ParseMode.HTML)
            return

        raw_nicks = subcommand.split(",")
        self.state.set_nicknames(raw_nicks)
        updated = self.state.get_nicknames()
        if updated:
            nicks_formatted = ", ".join(updated)
            await update.effective_message.reply_text(
                f"✅ Updated nicknames: <code>{html.escape(nicks_formatted)}</code>\n"
                f"The bot will now react directly when addressed by any of these nicknames, its name, or its Telegram username.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.effective_message.reply_text("❌ No valid nicknames provided.", parse_mode=ParseMode.HTML)

    async def cmd_timezone(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        if not args:
            cur = self.state.get_timezone()
            now_str = self.state.get_current_time_str()
            await update.effective_message.reply_text(f"🕒 Timezone is: <code>{cur}</code> ({now_str})", parse_mode=ParseMode.HTML)
            return

        tz_str = args[0].strip()
        if self.state.set_timezone(tz_str):
            await update.effective_message.reply_text(f"✅ Timezone updated to: <code>{tz_str}</code> ({self.state.get_current_time_str()})", parse_mode=ParseMode.HTML)
        else:
            await update.effective_message.reply_text(f"❌ Invalid timezone: <code>{tz_str}</code>. Use IANA format like <code>Europe/Budapest</code> or <code>UTC</code>.", parse_mode=ParseMode.HTML)

    async def cmd_sleep(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        if not args:
            sched = self.state.get_sleep_schedule()
            is_sleep = "Yes" if self.state.is_sleeping() else "No"
            status = f"{'ON' if sched.get('enabled') else 'OFF'} ({sched.get('sleep_start')} - {sched.get('sleep_end')}) (Sleeping: {is_sleep})"
            await update.effective_message.reply_text(f"🌙 Sleep schedule: <b>{status}</b>", parse_mode=ParseMode.HTML)
            return

        mode = args[0].lower()
        if mode == "off":
            self.state.set_sleep_schedule(enabled=False)
            self.schedule_spontaneous_job(context)
            await update.effective_message.reply_text("✅ Sleep schedule turned <b>OFF</b>.", parse_mode=ParseMode.HTML)
        elif mode == "on":
            start = args[1] if len(args) > 1 else None
            end = args[2] if len(args) > 2 else None
            self.state.set_sleep_schedule(enabled=True, sleep_start=start, sleep_end=end)
            self.schedule_spontaneous_job(context)
            sched = self.state.get_sleep_schedule()
            await update.effective_message.reply_text(f"✅ Sleep schedule enabled: <b>{sched.get('sleep_start')} to {sched.get('sleep_end')}</b>", parse_mode=ParseMode.HTML)
        else:
            await update.effective_message.reply_text("❌ Usage: <code>/sleep [on|off] [HH:MM] [HH:MM]</code>", parse_mode=ParseMode.HTML)

    async def cmd_spontaneous(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        spont = self.state.get_spontaneous_settings()
        min_h = spont.get("min_hours", 2.0)
        max_h = spont.get("max_hours", 4.0)
        if not args:
            status = "ON" if spont.get("enabled") else "OFF"
            await update.effective_message.reply_text(
                f"🎲 Spontaneous revival messages: <b>{status}</b>\n"
                f"• Interval: <b>{min_h:g} - {max_h:g} hours</b> (random)\n\n"
                f"Usage:\n"
                f"• <code>/spontaneous on</code>\n"
                f"• <code>/spontaneous off</code>\n"
                f"• <code>/spontaneous now</code>\n"
                f"• <code>/spontaneous_interval &lt;min_hours&gt; &lt;max_hours&gt;</code>",
                parse_mode=ParseMode.HTML,
            )
            return

        # Direct numeric shortcut: e.g. /spontaneous 2 4
        if len(args) == 2:
            try:
                n_min = float(args[0])
                n_max = float(args[1])
                await self._handle_set_spontaneous_interval(update, context, n_min, n_max)
                return
            except ValueError:
                pass

        mode = args[0].lower()
        if mode in ("now", "trigger", "send"):
            await self.cmd_spontaneous_now(update, context)
            return
        elif mode in ("interval", "range", "timer") and len(args) >= 3:
            try:
                n_min = float(args[1])
                n_max = float(args[2])
                await self._handle_set_spontaneous_interval(update, context, n_min, n_max)
                return
            except ValueError:
                await update.effective_message.reply_text("❌ Please specify valid numbers for min and max hours.", parse_mode=ParseMode.HTML)
                return
        elif mode == "off":
            self.state.set_spontaneous_settings(enabled=False)
            if context.job_queue:
                for job in context.job_queue.get_jobs_by_name("spontaneous_revival"):
                    job.schedule_removal()
            await update.effective_message.reply_text("✅ Spontaneous revival messages turned <b>OFF</b>.", parse_mode=ParseMode.HTML)
        elif mode == "on":
            self.state.set_spontaneous_settings(enabled=True)
            self.schedule_spontaneous_job(context)
            spont = self.state.get_spontaneous_settings()
            min_h = spont.get("min_hours", 2.0)
            max_h = spont.get("max_hours", 4.0)
            await update.effective_message.reply_text(
                f"✅ Spontaneous revival enabled (every <b>{min_h:g} - {max_h:g} hours</b>, random).",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.effective_message.reply_text(
                "❌ Usage: <code>/spontaneous [on|off|now]</code> or <code>/spontaneous_interval &lt;min_hours&gt; &lt;max_hours&gt;</code>",
                parse_mode=ParseMode.HTML,
            )

    async def _handle_set_spontaneous_interval(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, min_hours: float, max_hours: float
    ) -> None:
        if min_hours <= 0 or max_hours <= 0:
            await update.effective_message.reply_text("❌ Interval values must be positive numbers greater than 0.", parse_mode=ParseMode.HTML)
            return
        if min_hours > max_hours:
            await update.effective_message.reply_text("❌ Minimum hours cannot be greater than maximum hours.", parse_mode=ParseMode.HTML)
            return

        self.state.set_spontaneous_interval(min_hours, max_hours)
        if self.state.is_spontaneous_enabled():
            self.schedule_spontaneous_job(context)
        await update.effective_message.reply_text(
            f"✅ Spontaneous message interval updated: <b>{min_hours:g} - {max_hours:g} hours</b>.",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_spontaneous_interval(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        spont = self.state.get_spontaneous_settings()
        min_h = spont.get("min_hours", 2.0)
        max_h = spont.get("max_hours", 4.0)
        if len(args) < 2:
            await update.effective_message.reply_text(
                f"⚙️ Current spontaneous interval: <b>{min_h:g} to {max_h:g} hours</b> (random)\n\n"
                f"Usage: <code>/spontaneous_interval &lt;min_hours&gt; &lt;max_hours&gt;</code>\n"
                f"Example: <code>/spontaneous_interval 2 4</code>",
                parse_mode=ParseMode.HTML,
            )
            return

        try:
            n_min = float(args[0])
            n_max = float(args[1])
        except ValueError:
            await update.effective_message.reply_text("❌ Please specify valid numbers for min and max hours (e.g. <code>/spontaneous_interval 2 4</code>).", parse_mode=ParseMode.HTML)
            return

        await self._handle_set_spontaneous_interval(update, context, n_min, n_max)

    async def cmd_spontaneous_now(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        status_msg = await update.effective_message.reply_text("⏳ Generating spontaneous message for the group...")
        success, detail = await self.trigger_spontaneous_message(context)
        if success:
            await status_msg.edit_text(
                f"✅ <b>Spontaneous message dispatched to group:</b>\n\n{html.escape(detail)}",
                parse_mode=ParseMode.HTML,
            )
        else:
            await status_msg.edit_text(
                f"❌ Failed to dispatch spontaneous message: {html.escape(detail)}",
                parse_mode=ParseMode.HTML,
            )

    async def _process_prompt_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE, document: Any) -> None:
        user_id = update.effective_user.id if update.effective_user else None
        if not document:
            await update.effective_message.reply_text("❌ No document attached. Please upload a prompt text file.")
            return

        file_name = document.file_name or ""
        if file_name and not any(file_name.lower().endswith(ext) for ext in (".txt", ".md", ".prompt")):
            logger.warning("Uploaded prompt file '%s' does not have a text extension", file_name)

        try:
            file_obj = await context.bot.get_file(document.file_id)
            buf = io.BytesIO()
            await file_obj.download_to_memory(buf)
            raw_bytes = buf.getvalue()
        except Exception as e:
            logger.error("Failed to download prompt document from Telegram: %s", e)
            await update.effective_message.reply_text(f"❌ Failed to download file: {html.escape(str(e))}")
            return

        try:
            content_str = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            await update.effective_message.reply_text(f"❌ File must be UTF-8 encoded text: {html.escape(str(e))}")
            return

        content_clean = content_str.strip()
        if not content_clean:
            await update.effective_message.reply_text("❌ Prompt file cannot be empty.")
            return

        self.state.set_system_prompt(content_clean)
        if user_id:
            self._waiting_for_prompt_upload.discard(user_id)

        await update.effective_message.reply_text(
            f"✅ <b>Successfully loaded new system prompt!</b>\n"
            f"Length: {len(content_clean)} characters.",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        user_id = update.effective_user.id
        args = context.args or []
        subcommand = args[0].strip().lower() if args else ""

        if subcommand == "load":
            if update.effective_message.document:
                await self._process_prompt_upload(update, context, update.effective_message.document)
                return
            if user_id:
                self._waiting_for_prompt_upload.add(user_id)
            await update.effective_message.reply_text(
                "📥 <b>Upload System Prompt</b>\n"
                "Please upload the new prompt <code>.txt</code> file as a document attachment.\n\n"
                "The bot will check it for validity and replace current system prompt.\n"
                "Send /cancel to abort.",
                parse_mode=ParseMode.HTML,
            )
            return

        if subcommand == "reset":
            self.state.reset_system_prompt()
            await update.effective_message.reply_text("✅ System prompt reset to default persona.", parse_mode=ParseMode.HTML)
            return

        if subcommand:
            await update.effective_message.reply_text(
                "❌ Usage: <code>/prompt</code> (download file), <code>/prompt load</code> (upload new file), or <code>/prompt reset</code> (reset to default)",
                parse_mode=ParseMode.HTML,
            )
            return

        # No args: upload current base system prompt as a document file
        prompt = self.state.get_system_prompt()
        buf = io.BytesIO(prompt.encode("utf-8"))
        buf.name = "prompt.txt"
        await update.effective_message.reply_document(
            document=buf,
            filename="prompt.txt",
            caption=(
                "📝 <b>System Prompt</b>\n"
                "Edit this file and upload it with <code>/prompt load</code> to update the system prompt."
            ),
            parse_mode=ParseMode.HTML,
        )
    async def cmd_grounding(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        if not args:
            cur = "ON" if self.state.is_search_grounding_active() else "OFF"
            await update.effective_message.reply_text(f"🔍 Google Search Grounding is currently: <b>{cur}</b>", parse_mode=ParseMode.HTML)
            return

        mode = args[0].lower()
        if mode == "on":
            self.state.set_search_grounding_active(True)
            await update.effective_message.reply_text("✅ Google Search Grounding turned <b>ON</b>.", parse_mode=ParseMode.HTML)
        elif mode == "off":
            self.state.set_search_grounding_active(False)
            await update.effective_message.reply_text("✅ Google Search Grounding turned <b>OFF</b>.", parse_mode=ParseMode.HTML)
        else:
            await update.effective_message.reply_text("❌ Usage: <code>/grounding [on|off]</code>", parse_mode=ParseMode.HTML)

    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        if not args:
            cur = "ON" if self.state.is_debug_mode() else "OFF"
            await update.effective_message.reply_text(f"🐛 Debug mode is currently: <b>{cur}</b>", parse_mode=ParseMode.HTML)
            return

        mode = args[0].lower()
        if mode == "on":
            self.state.set_debug_mode(True)
            logging.getLogger().setLevel(logging.DEBUG)
            await update.effective_message.reply_text("✅ Debug mode turned <b>ON</b>. Raw LLM requests/responses and Telegram group messages will be printed to stdout.", parse_mode=ParseMode.HTML)
        elif mode == "off":
            self.state.set_debug_mode(False)
            logging.getLogger().setLevel(logging.INFO)
            await update.effective_message.reply_text("✅ Debug mode turned <b>OFF</b>.", parse_mode=ParseMode.HTML)
        else:
            await update.effective_message.reply_text("❌ Usage: <code>/debug [on|off]</code>", parse_mode=ParseMode.HTML)
    async def cmd_image_large_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        if not args:
            cur = "ON" if self.state.is_image_interpretation_large_model() else "OFF"
            desc = "Large model is used instantly for all image interpretations." if cur == "ON" else "Small model is tried first for image interpretation (with automatic escalation)."
            await update.effective_message.reply_text(
                f"🖼 Large Model for Image Interpretation is currently: <b>{cur}</b>\n<i>({desc})</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        mode = args[0].lower()
        if mode == "on":
            self.state.set_image_interpretation_large_model(True)
            await update.effective_message.reply_text(
                "✅ Large model for image interpretation turned <b>ON</b>. Images will be processed instantly with the large model.",
                parse_mode=ParseMode.HTML,
            )
        elif mode == "off":
            self.state.set_image_interpretation_large_model(False)
            await update.effective_message.reply_text(
                "✅ Large model for image interpretation turned <b>OFF</b>. Images will be interpreted using the small model first (with automatic escalation if needed).",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.effective_message.reply_text(
                "❌ Usage: <code>/image_large [on|off]</code>",
                parse_mode=ParseMode.HTML,
            )


    async def _process_memory_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE, document: Any) -> None:
        user_id = update.effective_user.id if update.effective_user else None
        if not document:
            await update.effective_message.reply_text("❌ No document attached. Please upload a JSON file.")
            return

        file_name = document.file_name or ""
        if file_name and not file_name.lower().endswith(".json"):
            logger.warning("Uploaded file '%s' does not have a .json extension", file_name)

        try:
            file_obj = await context.bot.get_file(document.file_id)
            buf = io.BytesIO()
            await file_obj.download_to_memory(buf)
            raw_bytes = buf.getvalue()
        except Exception as e:
            logger.error("Failed to download memory document from Telegram: %s", e)
            await update.effective_message.reply_text(f"❌ Failed to download file: {html.escape(str(e))}")
            return

        try:
            content_str = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            await update.effective_message.reply_text(f"❌ File must be UTF-8 encoded text: {html.escape(str(e))}")
            return

        try:
            parsed = json.loads(content_str)
        except json.JSONDecodeError as e:
            await update.effective_message.reply_text(f"❌ Invalid JSON format: {html.escape(str(e))}")
            return

        valid, err_msg, cleaned = validate_memory_dict(parsed)
        if not valid or cleaned is None:
            await update.effective_message.reply_text(f"❌ Invalid memories JSON structure: {html.escape(err_msg)}")
            return

        backup_file = self.memory.create_backup()
        self.memory.data = cleaned
        self.memory.save()
        self.memory.load()
        if user_id:
            self._waiting_for_memory_upload.discard(user_id)

        facts_count = len(cleaned.get("memories", []))
        dyn_count = len(cleaned.get("dynamics", []))
        jokes_count = len(cleaned.get("inside_jokes", []))
        bak_note = f"\n📦 Backup created: <code>{html.escape(os.path.basename(backup_file))}</code>" if backup_file else ""
        await update.effective_message.reply_text(
            f"✅ <b>Successfully validated and loaded new memories JSON!</b>\n"
            f"• Facts: {facts_count}\n"
            f"• Group Dynamics: {dyn_count}\n"
            f"• Inside Jokes: {jokes_count}"
            f"{bak_note}",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_memories(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        user_id = update.effective_user.id
        args = context.args or []
        subcommand = args[0].strip().lower() if args else ""

        if subcommand == "load":
            if update.effective_message.document:
                await self._process_memory_upload(update, context, update.effective_message.document)
                return
            if user_id:
                self._waiting_for_memory_upload.add(user_id)
            await update.effective_message.reply_text(
                "📥 <b>Upload Memories JSON</b>\n"
                "Please upload the new memories <code>.json</code> file as a document attachment.\n\n"
                "The bot will check it for validity and replace current memories.\n"
                "Send /cancel to abort.",
                parse_mode=ParseMode.HTML,
            )
            return

        if subcommand:
            await update.effective_message.reply_text(
                "❌ Usage: <code>/memories</code> (download file) or <code>/memories load</code> (upload new JSON file)",
                parse_mode=ParseMode.HTML,
            )
            return

        # No args: upload the memories JSON file as-is to the admin
        self.memory.save()
        if not os.path.exists(self.memory.file_path):
            await update.effective_message.reply_text("❌ Memory file does not exist on disk.")
            return

        with open(self.memory.file_path, "rb") as f:
            await update.effective_message.reply_document(
                document=f,
                filename=os.path.basename(self.memory.file_path),
                caption=(
                    "🧠 <b>Memories JSON</b>\n"
                    "Edit this file and upload it with <code>/memories load</code> to update memories."
                ),
                parse_mode=ParseMode.HTML,
            )

    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return
        user_id = update.effective_user.id
        cancelled = False
        if user_id and user_id in self._waiting_for_memory_upload:
            self._waiting_for_memory_upload.discard(user_id)
            cancelled = True
        if user_id and user_id in self._waiting_for_prompt_upload:
            self._waiting_for_prompt_upload.discard(user_id)
            cancelled = True

        if cancelled:
            await update.effective_message.reply_text("❌ Upload cancelled.")
        else:
            await update.effective_message.reply_text("Nothing to cancel.")
    async def cmd_curate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        msg = await update.effective_message.reply_text("⏳ Consolidating memories from recent conversation...")
        summary = await self.trigger_curation()
        text = (
            f"✅ <b>Memory Consolidation Complete:</b>\n"
            f"• Facts Added: {summary['added_facts']}\n"
            f"• Facts Updated: {summary['updated_facts']}\n"
            f"• Facts Discarded: {summary['discarded_facts']}\n"
            f"• Dynamics Added: {summary['added_dynamics']}\n"
            f"• Dynamics Discarded: {summary['discarded_dynamics']}\n"
            f"• Jokes Added: {summary['added_jokes']}\n"
            f"• Jokes Discarded: {summary['discarded_jokes']}"
        )
        await msg.edit_text(text, parse_mode=ParseMode.HTML)

    def register(self, application: Application) -> None:
        """Registers all command and message handlers with the Telegram application."""
        # Admin slash commands
        application.add_handler(CommandHandler(["start", "help"], self.cmd_start_or_help))
        application.add_handler(CommandHandler("status", self.cmd_status))
        application.add_handler(CommandHandler("talkativeness", self.cmd_talkativeness))
        application.add_handler(CommandHandler("cooldown", self.cmd_cooldown))
        application.add_handler(CommandHandler("language", self.cmd_language))
        application.add_handler(CommandHandler("timezone", self.cmd_timezone))
        application.add_handler(CommandHandler("nicknames", self.cmd_nicknames))
        application.add_handler(CommandHandler("sleep", self.cmd_sleep))
        application.add_handler(CommandHandler("spontaneous", self.cmd_spontaneous))
        application.add_handler(CommandHandler(["spontaneous_now", "spontaneousnow"], self.cmd_spontaneous_now))
        application.add_handler(CommandHandler(["spontaneous_interval", "spontaneousinterval"], self.cmd_spontaneous_interval))
        application.add_handler(CommandHandler("prompt", self.cmd_prompt))
        application.add_handler(CommandHandler("grounding", self.cmd_grounding))
        application.add_handler(CommandHandler("debug", self.cmd_debug))
        application.add_handler(CommandHandler(["image_large", "image_large_model", "imagelarge"], self.cmd_image_large_model))
        application.add_handler(CommandHandler("memories", self.cmd_memories))
        application.add_handler(CommandHandler("cancel", self.cmd_cancel))
        application.add_handler(CommandHandler("curate", self.cmd_curate))

        # Main message handler (captures text, photos, media in groups and private)
        application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, self.on_message))
