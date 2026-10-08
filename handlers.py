import asyncio
import base64
import difflib
import html
import io
import json
import logging
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import uuid

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta

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
/scheduled [list|cancel <id>] - View or cancel active scheduled replies and reminders
<b>Prompt, Grounding & Vision</b>
/prompt - Upload system prompt text file as-is
/prompt load - Expect a system prompt text file upload to validate and load
/prompt reset - Reset to default persona prompt
/grounding [on|off] - Toggle Google Search grounding for real-time web info
/search_small [on|off] - Toggle small model direct web search (ON: small searches directly if capable, OFF: delegates to large model)
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


# Consecutive-duplicate protection: the bot must never post the same message twice in a row.
REPEAT_SIMILARITY_THRESHOLD = 0.9
REPEAT_FUZZY_MIN_CHARS = 8
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_message_text(text: str) -> str:
    """Normalizes message text for consecutive-duplicate detection (tags, entities, whitespace, case)."""
    plain = html.unescape(_HTML_TAG_PATTERN.sub(" ", text or ""))
    return _WHITESPACE_PATTERN.sub(" ", plain).strip().casefold()


def is_repeated_message(previous_text: str, candidate_text: str) -> bool:
    """True when `candidate_text` is the same or nearly identical to `previous_text`."""
    previous = normalize_message_text(previous_text)
    candidate = normalize_message_text(candidate_text)
    if not previous or not candidate:
        return False
    if previous == candidate:
        return True
    if min(len(previous), len(candidate)) < REPEAT_FUZZY_MIN_CHARS:
        return False
    matcher = difflib.SequenceMatcher(None, previous, candidate)
    if matcher.quick_ratio() < REPEAT_SIMILARITY_THRESHOLD:
        return False
    return matcher.ratio() >= REPEAT_SIMILARITY_THRESHOLD


INTERVAL_UNIT_MAP = {
    "y": "years", "yr": "years", "yrs": "years", "year": "years", "years": "years",
    "mo": "months", "month": "months", "months": "months",
    "w": "weeks", "wk": "weeks", "wks": "weeks", "week": "weeks", "weeks": "weeks",
    "d": "days", "day": "days", "days": "days",
    "h": "hours", "hr": "hours", "hrs": "hours", "hour": "hours", "hours": "hours",
    "m": "minutes", "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
    "s": "seconds", "sec": "seconds", "secs": "seconds", "second": "seconds", "seconds": "seconds",
}

def parse_interval_relativedelta(interval_str: str, min_seconds: int = 60) -> Optional[relativedelta]:
    if not interval_str:
        return None
    s = interval_str.strip()
    matches = list(re.finditer(r"(\d+)\s*([a-zA-Z]+)", s))
    if not matches:
        return None
    kwargs = {"years": 0, "months": 0, "weeks": 0, "days": 0, "hours": 0, "minutes": 0, "seconds": 0}
    matched_any = False
    for m in matches:
        unit = m.group(2).lower()
        if unit not in INTERVAL_UNIT_MAP:
            return None
        val = int(m.group(1))
        kwargs[INTERVAL_UNIT_MAP[unit]] += val
        matched_any = True
    if not matched_any:
        return None
    if (
        min_seconds > 0
        and kwargs["years"] == 0
        and kwargs["months"] == 0
        and kwargs["weeks"] == 0
        and kwargs["days"] == 0
        and kwargs["hours"] == 0
        and kwargs["minutes"] == 0
        and kwargs["seconds"] < min_seconds
    ):
        kwargs["seconds"] = min_seconds
    return relativedelta(**kwargs)

def serialize_interval(rd: Optional[relativedelta]) -> Optional[Dict[str, int]]:
    if rd is None:
        return None
    return {
        "years": int(getattr(rd, "years", 0)),
        "months": int(getattr(rd, "months", 0)),
        "days": int(getattr(rd, "days", 0)),
        "hours": int(getattr(rd, "hours", 0)),
        "minutes": int(getattr(rd, "minutes", 0)),
        "seconds": int(getattr(rd, "seconds", 0)),
    }

def deserialize_interval(d: Optional[Dict[str, Any]]) -> relativedelta:
    if not d or not isinstance(d, dict):
        return relativedelta(seconds=60)
    return relativedelta(
        years=int(d.get("years", 0)),
        months=int(d.get("months", 0)),
        days=int(d.get("days", 0)),
        hours=int(d.get("hours", 0)),
        minutes=int(d.get("minutes", 0)),
        seconds=int(d.get("seconds", 0)),
    )

def parse_schedule_time(time_str: str, tz: Any, now: datetime) -> Optional[datetime]:
    if not time_str:
        return None
    raw = time_str.strip()

    # 1. Relative offset e.g. "in 3 days", "+2h", "in 30m", "+ 1 month"
    rel_clean = re.sub(r"^(?:in|\+)\s*", "", raw, flags=re.IGNORECASE).strip()
    rd = parse_interval_relativedelta(rel_clean, min_seconds=1)
    if rd is not None:
        return now + rd

    # 2. Time-only format: HH:MM or HH:MM:SS
    m_time = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", raw)
    if m_time:
        hr = int(m_time.group(1))
        mn = int(m_time.group(2))
        sec = int(m_time.group(3) or 0)
        if 0 <= hr < 24 and 0 <= mn < 60 and 0 <= sec < 60:
            target_dt = now.replace(hour=hr, minute=mn, second=sec, microsecond=0)
            if target_dt <= now:
                target_dt += relativedelta(days=1)
            return target_dt

    # 3. Absolute timestamp via dateutil parser
    try:
        parsed = dateutil_parser.parse(raw)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=tz)
        return parsed.astimezone(tz)
    except Exception:
        return None


def _entry_age_dt(entry: Dict[str, Any], use_updated_at: bool = False) -> Optional[datetime]:
    """Parses an entry's age-basis timestamp as an aware UTC datetime; None when missing/unparseable."""
    if use_updated_at:
        ts = str(entry.get("updated_at") or entry.get("created_at") or "")
    else:
        ts = str(entry.get("created_at") or "")
    if not ts.strip():
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _compute_archive_candidates(hot_memories: Dict[str, Any], archive_days: int,
                                now: Optional[datetime] = None) -> Dict[str, List[str]]:
    """Lists hot entries older than `archive_days` in compact form for the curator prompt.

    Age basis: facts use updated_at if present else created_at; dynamics and jokes
    use created_at only. Entries with missing/unparseable timestamps are NEVER
    candidates (never archive what you cannot date).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = timedelta(days=archive_days)
    candidates: Dict[str, List[str]] = {"facts": [], "dynamics": [], "jokes": []}

    for m in hot_memories.get("memories", []):
        dt = _entry_age_dt(m, use_updated_at=True)
        if dt is not None and (now - dt) > cutoff:
            topic = str(m.get("topic", "")).strip()
            if topic:
                candidates["facts"].append(topic)

    for d in hot_memories.get("dynamics", []):
        dt = _entry_age_dt(d)
        if dt is not None and (now - dt) > cutoff:
            members = " & ".join(str(x) for x in d.get("members", []))
            if members:
                candidates["dynamics"].append(members)

    for j in hot_memories.get("inside_jokes", []):
        dt = _entry_age_dt(j)
        if dt is not None and (now - dt) > cutoff:
            title = str(j.get("title", "")).strip()
            if title:
                candidates["jokes"].append(title)

    return candidates


class BotHandlers:
    def __init__(self, params: Params, state: StateManager, memory: MemoryManager, llm: LLMClient,
                 deep_memory: Optional[MemoryManager] = None):
        self.params = params
        self.state = state
        self.memory = memory
        # Deep (cold-tier) memory. When not provided (legacy callers, unit
        # tests), fall back to an EMPTY in-memory manager bound to the
        # configured deep-memory path — never the hot file, and not loaded
        # from disk (it only persists if code explicitly saves it).
        if deep_memory is None:
            deep_path = getattr(params, "deepmemory_file", "")
            if not isinstance(deep_path, str) or not deep_path.strip():
                deep_path = "pletykas-deepmemory.json"
            deep_memory = MemoryManager(deep_path)
        self.deep_memory = deep_memory
        self.llm = llm
        self._debounce_jobs: Dict[int, Any] = {}
        self._active_evaluations: set[int] = set()
        self._waiting_for_memory_upload: set[int] = set()
        self._waiting_for_prompt_upload: set[int] = set()
        self._curation_lock = threading.Lock()
        self._next_spontaneous_time: Optional[datetime] = None
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

    # --- Consecutive Duplicate Guard ---

    def _get_last_bot_message_text(self, context: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
        """Returns the text of the most recent message this bot itself posted to the group."""
        bot_id = getattr(context.bot, "id", None)
        if bot_id is None:
            return None
        for item in reversed(self.state.get_chat_history()):
            if item.get("from_user_id") == bot_id:
                return item.get("text") or ""
        return None

    def _suppress_repeated_reply(self, text: str, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Returns True when `text` repeats the bot's own previous group message and must not be sent."""
        previous = self._get_last_bot_message_text(context)
        if previous is None or not is_repeated_message(previous, text):
            return False
        preview = normalize_message_text(text)[:120]
        logger.warning("Suppressed repeated bot message (identical/near-identical to previous group message): %s", preview)
        if self.state.is_debug_mode():
            self._log_debug_group_msg("SUPPRESSED", f"Duplicate reply not sent (matches previous bot message): {preview}")
        return True


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

    def schedule_spontaneous_job(self, context: ContextTypes.DEFAULT_TYPE, force_new: bool = False) -> None:
        if not self.state.is_spontaneous_enabled():
            if context.job_queue:
                for job in context.job_queue.get_jobs_by_name("spontaneous_revival"):
                    job.schedule_removal()
            self._next_spontaneous_time = None
            self.state.set_spontaneous_next_fire_time(None)
            return

        if not context.job_queue:
            logger.debug("JobQueue not initialized, cannot schedule spontaneous job")
            self._next_spontaneous_time = None
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

        now = datetime.now(self.state.get_tzinfo())
        persisted_dt = self.state.get_spontaneous_next_fire_time() if not force_new else None

        if persisted_dt and persisted_dt > now:
            delay_sec = (persisted_dt - now).total_seconds()
            target_dt = persisted_dt
            logger.info("Resuming scheduled spontaneous revival in %.2f hours (%.0f seconds)", delay_sec / 3600.0, delay_sec)
        else:
            delay_sec = random.uniform(min_hours * 3600.0, max_hours * 3600.0)
            target_dt = now + timedelta(seconds=delay_sec)
            self.state.set_spontaneous_next_fire_time(target_dt)
            logger.info("Scheduled new spontaneous revival in %.2f hours (%.0f seconds)", delay_sec / 3600.0, delay_sec)

        self._next_spontaneous_time = target_dt

        context.job_queue.run_once(
            self._spontaneous_callback,
            when=delay_sec,
            name="spontaneous_revival",
            data={"group_chat_id": self.params.group_chat_id},
        )
    def get_next_spontaneous_time(self, context: Optional[ContextTypes.DEFAULT_TYPE] = None) -> Optional[datetime]:
        """Returns the datetime of the next scheduled spontaneous revival message, if any."""
        if not self.state.is_spontaneous_enabled():
            return None
        if context and context.job_queue:
            jobs = context.job_queue.get_jobs_by_name("spontaneous_revival")
            for job in jobs:
                if getattr(job, "removed", False):
                    continue
                try:
                    next_t = job.next_t
                    if next_t:
                        return next_t
                except Exception:
                    pass
        return getattr(self, "_next_spontaneous_time", None)
    async def _spontaneous_callback(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Triggered periodically at random intervals to post spontaneous messages."""
        if self.state.is_sleeping():
            logger.info("Spontaneous trigger fired during sleep hours; skipping and rescheduling.")
            self.state.set_spontaneous_next_fire_time(None)
            self.schedule_spontaneous_job(context)
            return

        await self.trigger_spontaneous_message(context)
        self.state.set_spontaneous_next_fire_time(None)
        self.schedule_spontaneous_job(context)

    async def _send_group_poll(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        question: str,
        options: List[str],
        reply_to_msg_id: Optional[int] = None,
        poll_type_label: str = "Poll",
    ) -> Optional[Any]:
        """Sends a Telegram poll to the group, logs debug output, and records it in history."""
        sent = None
        try:
            if reply_to_msg_id:
                try:
                    sent = await context.bot.send_poll(
                        chat_id=chat_id,
                        question=question,
                        options=options,
                        is_anonymous=False,
                        reply_to_message_id=reply_to_msg_id,
                    )
                except Exception as re:
                    logger.debug("Failed sending poll with reply_to_message_id (%s), sending without reply", re)
                    sent = await context.bot.send_poll(
                        chat_id=chat_id,
                        question=question,
                        options=options,
                        is_anonymous=False,
                    )
            else:
                sent = await context.bot.send_poll(
                    chat_id=chat_id,
                    question=question,
                    options=options,
                    is_anonymous=False,
                )
        except Exception as e:
            logger.error("Failed to send %s to chat %s: %s", poll_type_label, chat_id, e)
            return None

        now_str = self.state.get_current_time_str()
        bot_name = getattr(context.bot, "first_name", None) or "Pletykas"
        bot_name = str(bot_name) if isinstance(bot_name, str) else "Pletykas"
        raw_bot_id = getattr(context.bot, "id", 0)
        bot_id = int(raw_bot_id) if isinstance(raw_bot_id, (int, float)) else 0

        if self.state.is_debug_mode():
            reply_info = f" | Reply to msg ID: {reply_to_msg_id}" if reply_to_msg_id else ""
            self._log_debug_group_msg(
                "OUTGOING",
                f"Chat ID: {chat_id} | Message ID: {sent.message_id}{reply_info}\nType: {poll_type_label}\nQuestion: {question}\nOptions: {', '.join(options)}",
            )

        poll_text = f"[Poll: {question}] Options: {', '.join(options)}"
        entry = {
            "id": sent.message_id,
            "from_user_id": bot_id,
            "from_user_name": bot_name,
            "timestamp_epoch": time.time(),
            "timestamp_str": now_str,
            "reply_to_msg_id": reply_to_msg_id,
            "reply_to_user_name": None,
            "text": poll_text,
            "media_type": "poll",
            "media_b64": None,
        }
        self.state.append_chat_message(entry)
        self.state.append_memory_message(entry)
        return sent

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

        cleaned_text, poll_spec = self.llm.extract_poll(response)
        suppressed_repeat = bool(cleaned_text) and self._suppress_repeated_reply(cleaned_text, context)
        if suppressed_repeat:
            cleaned_text = None

        sent_any = False
        sent_texts: List[str] = []

        if cleaned_text:
            try:
                try:
                    sent = await context.bot.send_message(chat_id=group_id, text=cleaned_text, parse_mode=ParseMode.HTML)
                except Exception as pe:
                    logger.debug("Failed sending spontaneous message with HTML parse mode (%s), falling back to plain text", pe)
                    sent = await context.bot.send_message(chat_id=group_id, text=cleaned_text)
                now_str = self.state.get_current_time_str()
                bot_name = getattr(context.bot, "first_name", None) or "Pletykas"
                bot_name = str(bot_name) if isinstance(bot_name, str) else "Pletykas"
                raw_bot_id = getattr(context.bot, "id", 0)
                bot_id = int(raw_bot_id) if isinstance(raw_bot_id, (int, float)) else 0
                if self.state.is_debug_mode():
                    self._log_debug_group_msg(
                        "OUTGOING",
                        f"Chat ID: {group_id} | Message ID: {sent.message_id}\nType: Spontaneous Text Message\nText: {cleaned_text}",
                    )
                entry = {
                    "id": sent.message_id,
                    "from_user_id": bot_id,
                    "from_user_name": bot_name,
                    "timestamp_epoch": time.time(),
                    "timestamp_str": now_str,
                    "reply_to_msg_id": None,
                    "reply_to_user_name": None,
                    "text": cleaned_text,
                    "media_type": "none",
                    "media_b64": None,
                }
                self.state.append_chat_message(entry)
                self.state.append_memory_message(entry)
                sent_any = True
                sent_texts.append(cleaned_text)
            except Exception as e:
                logger.error("Failed to send spontaneous message: %s", e)
                if not poll_spec:
                    return False, f"Failed to send message: {e}"

        if poll_spec:
            sent_poll = await self._send_group_poll(
                context=context,
                chat_id=group_id,
                question=poll_spec["question"],
                options=poll_spec["options"],
                reply_to_msg_id=None,
                poll_type_label="Spontaneous Poll",
            )
            if sent_poll:
                sent_any = True
                sent_texts.append(f"[Poll: {poll_spec['question']}] Options: {', '.join(poll_spec['options'])}")
            elif not sent_any:
                return False, "Failed to send spontaneous poll."

        if sent_any:
            return True, "\n\n".join(sent_texts)
        if suppressed_repeat:
            return False, "Skipped: the generated message repeated the previous bot message."
        return False, "LLM returned empty spontaneous message."
    # --- Scheduled Replies ---

    def schedule_reply_job(self, context: ContextTypes.DEFAULT_TYPE, entry: Dict[str, Any]) -> None:
        """Schedules a run_once job in JobQueue for the given schedule entry."""
        if not context.job_queue:
            logger.debug("JobQueue not initialized, cannot schedule reply job")
            return

        sched_id = entry.get("id")
        if not sched_id:
            return

        job_name = f"scheduled_reply_{sched_id}"
        # Remove any existing job with that name first
        for j in context.job_queue.get_jobs_by_name(job_name):
            j.schedule_removal()

        raw_target = entry.get("target_time")
        if not raw_target:
            return

        now = self.state.get_current_time()
        tz = self.state.get_tzinfo()
        try:
            target_dt = datetime.fromisoformat(raw_target)
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=tz)
            else:
                target_dt = target_dt.astimezone(tz)
        except Exception as e:
            logger.error("Failed to parse target_time for schedule '%s': %s", sched_id, e)
            return

        delay_sec = max(1.0, (target_dt - now).total_seconds())
        context.job_queue.run_once(
            self._scheduled_reply_callback,
            when=delay_sec,
            name=job_name,
            data={"schedule_id": sched_id},
        )
        logger.info("Scheduled reply job '%s' for %s (delay: %.1fs)", sched_id, target_dt.isoformat(), delay_sec)

    def cancel_reply_job(self, context: ContextTypes.DEFAULT_TYPE, schedule_id: str) -> bool:
        """Cancels a scheduled reply job in JobQueue by schedule ID."""
        if not context.job_queue:
            return False
        job_name = f"scheduled_reply_{schedule_id}"
        jobs = context.job_queue.get_jobs_by_name(job_name)
        found = False
        for j in jobs:
            j.schedule_removal()
            found = True
        return found

    async def _scheduled_reply_callback(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Invoked by JobQueue when a scheduled reply timer fires."""
        job = context.job
        schedule_id = job.data.get("schedule_id") if job and job.data else None
        if not schedule_id:
            logger.warning("Scheduled reply callback fired without schedule_id")
            return

        entry = self.state.get_scheduled_reply(schedule_id)
        if not entry:
            logger.warning("Scheduled reply entry '%s' not found in state (may have been cancelled)", schedule_id)
            return

        sched_type = entry.get("type", "oneshot")
        sched_desc = entry.get("description", "")
        chat_id = self.state.get_group_chat_id() or self.params.group_chat_id or entry.get("chat_id")
        target_msg_id = entry.get("target_msg_id")

        # Sleep schedule handling: periodic replies defer during quiet hours; oneshot execute regardless
        if sched_type == "periodic" and self.state.is_sleeping():
            logger.info("Periodic schedule '%s' triggered during quiet hours; advancing to next interval.", schedule_id)
            now = self.state.get_current_time()
            interval_rd = deserialize_interval(entry.get("interval_spec"))
            try:
                curr_target = datetime.fromisoformat(entry["target_time"])
                if curr_target.tzinfo is None:
                    curr_target = curr_target.replace(tzinfo=self.state.get_tzinfo())
            except Exception:
                curr_target = now
            next_dt = curr_target + interval_rd
            while next_dt <= now:
                next_dt += interval_rd
            self.state.update_scheduled_reply(schedule_id, {"target_time": next_dt.isoformat()})
            updated_entry = self.state.get_scheduled_reply(schedule_id)
            if updated_entry:
                self.schedule_reply_job(context, updated_entry)
            return

        recent_transcript = self._format_transcript(self.state.get_chat_history())
        memory_context = self.memory.format_for_context()
        sys_prompt = self.state.get_effective_system_prompt()
        tz_str = self.state.get_timezone()

        try:
            logger.info("Generating scheduled reply for job '%s' (type=%s, chat=%s)...", schedule_id, sched_type, chat_id)
            reply_text = await self.llm.generate_scheduled_reply(
                system_prompt=sys_prompt,
                memory_context=memory_context,
                transcript=recent_transcript,
                scheduled_description=sched_desc,
                scheduled_type=sched_type,
                timezone_str=tz_str,
            )

            if reply_text:
                reply_text, poll_spec = self.llm.extract_poll(reply_text)
                if reply_text and self._suppress_repeated_reply(reply_text, context):
                    reply_text = None
                sent_msg = None
                if reply_text:
                    try:
                        sent_msg = await context.bot.send_message(
                            chat_id=chat_id,
                            text=reply_text,
                            reply_to_message_id=target_msg_id,
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception as pe:
                        logger.debug("Failed sending scheduled reply with HTML parse mode (%s), attempting without HTML", pe)
                        try:
                            sent_msg = await context.bot.send_message(
                                chat_id=chat_id,
                                text=reply_text,
                                reply_to_message_id=target_msg_id,
                            )
                        except Exception as re:
                            logger.debug("Failed sending scheduled reply with reply_to_message_id (%s), sending to root", re)
                            try:
                                sent_msg = await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=reply_text,
                                )
                            except Exception as e_send:
                                logger.error("Failed to send scheduled reply message: %s", e_send)

                    if sent_msg:
                        now_str = self.state.get_current_time_str()
                        if self.state.is_debug_mode():
                            reply_info = f" | Reply to msg ID: {target_msg_id}" if target_msg_id else ""
                            self._log_debug_group_msg(
                                "OUTGOING",
                                f"Chat ID: {chat_id} | Message ID: {sent_msg.message_id}{reply_info}\nType: Scheduled Reply ({sched_type})\nText: {reply_text}",
                            )

                        bot_uid = getattr(context.bot, "id", 0)
                        bot_uname = getattr(context.bot, "first_name", "Pletykas")
                        msg_entry = {
                            "id": sent_msg.message_id,
                            "from_user_id": int(bot_uid) if isinstance(bot_uid, (int, float)) else 0,
                            "from_user_name": str(bot_uname) if isinstance(bot_uname, str) else "Pletykas",
                            "timestamp_epoch": time.time(),
                            "timestamp_str": now_str,
                            "reply_to_msg_id": target_msg_id,
                            "reply_to_user_name": None,
                            "text": reply_text,
                            "media_type": "none",
                            "media_b64": None,
                        }
                        self.state.append_chat_message(msg_entry)
                        self.state.append_memory_message(msg_entry)

                if poll_spec:
                    poll_reply_to = target_msg_id if not reply_text else None
                    await self._send_group_poll(
                        context=context,
                        chat_id=chat_id,
                        question=poll_spec["question"],
                        options=poll_spec["options"],
                        reply_to_msg_id=poll_reply_to,
                        poll_type_label=f"Scheduled Poll ({sched_type})",
                    )
        except Exception as e:
            logger.error("Error executing scheduled reply callback for '%s': %s", schedule_id, e, exc_info=True)

        # Post-execution lifecycle:
        if sched_type == "oneshot":
            self.state.remove_scheduled_reply(schedule_id)
            logger.info("Oneshot scheduled reply '%s' executed and removed from state", schedule_id)
        elif sched_type == "periodic":
            now = self.state.get_current_time()
            interval_rd = deserialize_interval(entry.get("interval_spec"))
            try:
                curr_target = datetime.fromisoformat(entry["target_time"])
                if curr_target.tzinfo is None:
                    curr_target = curr_target.replace(tzinfo=self.state.get_tzinfo())
            except Exception:
                curr_target = now
            next_dt = curr_target + interval_rd
            while next_dt <= now:
                next_dt += interval_rd
            self.state.update_scheduled_reply(schedule_id, {"target_time": next_dt.isoformat()})
            updated_entry = self.state.get_scheduled_reply(schedule_id)
            if updated_entry:
                self.schedule_reply_job(context, updated_entry)
            logger.info("Periodic scheduled reply '%s' rescheduled for %s", schedule_id, next_dt.isoformat())

    def load_and_schedule_pending_replies(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Loads and reschedules pending scheduled replies from persistent state upon bot startup."""
        if not context.job_queue:
            logger.warning("JobQueue not available, skipping scheduled replies restoration")
            return

        replies = self.state.get_scheduled_replies()
        total = len(replies)
        active_count = 0
        now = self.state.get_current_time()
        tz = self.state.get_tzinfo()

        for entry in replies:
            sched_id = entry.get("id")
            if not sched_id:
                continue
            sched_type = entry.get("type", "oneshot")
            raw_target = entry.get("target_time")
            if not raw_target:
                continue

            try:
                target_dt = datetime.fromisoformat(raw_target)
                if target_dt.tzinfo is None:
                    target_dt = target_dt.replace(tzinfo=tz)
                else:
                    target_dt = target_dt.astimezone(tz)
            except Exception as e:
                logger.warning("Failed to parse target_time for schedule '%s': %s", sched_id, e)
                continue

            if sched_type == "oneshot":
                if target_dt > now:
                    self.schedule_reply_job(context, entry)
                    active_count += 1
                else:
                    elapsed = (now - target_dt).total_seconds()
                    if elapsed < 900:  # < 15 minutes
                        logger.info("Oneshot schedule '%s' was due %.0fs ago (<15m). Firing immediately.", sched_id, elapsed)
                        job_name = f"scheduled_reply_{sched_id}"
                        context.job_queue.run_once(
                            self._scheduled_reply_callback,
                            when=1.0,
                            name=job_name,
                            data={"schedule_id": sched_id},
                        )
                        active_count += 1
                    else:
                        logger.warning("Oneshot schedule '%s' expired %.0fs ago (>=15m) while offline. Discarding.", sched_id, elapsed)
                        self.state.remove_scheduled_reply(sched_id)
            elif sched_type == "periodic":
                interval_rd = deserialize_interval(entry.get("interval_spec"))
                if target_dt > now:
                    self.schedule_reply_job(context, entry)
                    active_count += 1
                else:
                    while target_dt <= now:
                        target_dt += interval_rd
                    self.state.update_scheduled_reply(sched_id, {"target_time": target_dt.isoformat()})
                    entry["target_time"] = target_dt.isoformat()
                    self.schedule_reply_job(context, entry)
                    active_count += 1

        logger.info("Loaded %d scheduled replies from state file (%d active).", total, active_count)

    # --- Memory Curation ---

    def _curate_worker(self, blocking: bool = False) -> Dict[str, Any]:
        """Synchronous worker that runs reflection and curation of recent chat messages into permanent memory.
        Runs in a separate thread. Guarded by _curation_lock so only one curation job runs at a time.
        """
        acquired = self._curation_lock.acquire(blocking=blocking)
        if not acquired:
            logger.info("Memory curation already in progress in another thread; skipping concurrent trigger.")
            return {}

        try:
            mem_history = self.state.get_memory_history()
            transcript = self._format_transcript(mem_history)
            current_memories = self.memory.get_all_memories()

            # Age-based archive candidacy: 0 disables it (no candidates computed)
            archive_days = self.state.get_deep_archive_days()
            if archive_days > 0:
                archive_candidates = _compute_archive_candidates(current_memories, archive_days)
            else:
                archive_candidates = None
            deep_current = self.deep_memory.get_all_memories()

            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                curation = loop.run_until_complete(
                    self.llm.curate_memory(
                        current_memories,
                        transcript,
                        deep_memories=deep_current,
                        archive_candidates=archive_candidates,
                        archive_age_days=archive_days,
                    )
                )
            finally:
                try:
                    loop.run_until_complete(self.llm.close_thread_session())
                except Exception:
                    pass
                loop.close()

            # Create timestamped backup before committing updates (MemoryManager is thread-locked)
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

            # Deep-memory moves: archive hot -> deep, promote deep -> hot.
            # Cross-store moves are pop-then-append (two atomic saves). Holding both
            # managers' locks simultaneously is not possible by design, so a crash
            # between the two saves can lose at most the ONE entry being moved
            # (accepted: archived entries are low value). Each manager self-locks.
            self.deep_memory.create_backup()
            archived_facts = 0
            archived_dyn = 0
            archived_jokes = 0
            promoted_facts = 0
            promoted_dyn = 0
            promoted_jokes = 0

            for target in curation.get("facts_to_archive", []):
                entry = self.memory.pop_memory(target)
                if entry:
                    self.deep_memory.append_entry("memories", entry)
                    archived_facts += 1

            for target in curation.get("dynamics_to_archive", []):
                entry = self.memory.pop_dynamic(target)
                if entry:
                    self.deep_memory.append_entry("dynamics", entry)
                    archived_dyn += 1

            for target in curation.get("jokes_to_archive", []):
                entry = self.memory.pop_inside_joke(target)
                if entry:
                    self.deep_memory.append_entry("inside_jokes", entry)
                    archived_jokes += 1

            for target in curation.get("facts_to_promote", []):
                entry = self.deep_memory.pop_memory(target)
                if entry:
                    self.memory.append_entry("memories", entry)
                    promoted_facts += 1

            for target in curation.get("dynamics_to_promote", []):
                entry = self.deep_memory.pop_dynamic(target)
                if entry:
                    self.memory.append_entry("dynamics", entry)
                    promoted_dyn += 1

            for target in curation.get("jokes_to_promote", []):
                entry = self.deep_memory.pop_inside_joke(target)
                if entry:
                    self.memory.append_entry("inside_jokes", entry)
                    promoted_jokes += 1

            # Forced-eviction fallback: hard cap against pathological hot-fact growth
            # (count-based, intentionally far above any healthy hot set). Oldest
            # facts by the same age basis move to deep; undateable entries count as
            # oldest. Dynamics/jokes have no forced eviction (their volume is low).
            HOT_FACT_HARD_CAP = 300
            if len(self.memory.get_all_memories()["memories"]) > HOT_FACT_HARD_CAP:
                logger.warning(
                    "Hot memory exceeds %d facts; force-evicting oldest facts to deep memory.",
                    HOT_FACT_HARD_CAP,
                )
                while len(self.memory.get_all_memories()["memories"]) > HOT_FACT_HARD_CAP:
                    facts = self.memory.get_all_memories()["memories"]
                    oldest = min(
                        facts,
                        key=lambda f: _entry_age_dt(f, use_updated_at=True) or datetime.min.replace(tzinfo=timezone.utc),
                    )
                    entry = self.memory.pop_memory({
                        "topic": oldest.get("topic", ""),
                        "content": oldest.get("content", ""),
                    })
                    if not entry:
                        logger.warning("Forced eviction could not pop the oldest fact; stopping eviction.")
                        break
                    self.deep_memory.append_entry("memories", entry)
                    archived_facts += 1

            # Phase 3 hook: refresh deep-memory embedding vectors here (after moves).

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
                "archived_facts": archived_facts,
                "archived_dynamics": archived_dyn,
                "archived_jokes": archived_jokes,
                "promoted_facts": promoted_facts,
                "promoted_dynamics": promoted_dyn,
                "promoted_jokes": promoted_jokes,
            }
            logger.info("Memory consolidation completed in worker thread: %s", summary)
            return summary
        except Exception as e:
            logger.error("Error executing history curation worker: %s", e, exc_info=True)
            return {}
        finally:
            self._curation_lock.release()

    def trigger_curation_background(self) -> None:
        """Spawns history curation in a separate background thread without blocking the event loop."""
        t = threading.Thread(
            target=self._curate_worker,
            args=(False,),
            daemon=True,
            name="history-curation",
        )
        t.start()

    async def trigger_curation(self, blocking: bool = True) -> Dict[str, Any]:
        """Runs reflection and curation of recent chat messages into permanent memory in a worker thread."""
        return await asyncio.to_thread(self._curate_worker, blocking)
    # --- Message Helpers & Formatting ---

    def _format_transcript(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return "(No recent messages)"
        lines = []
        for msg in history:
            reply_part = f" (replying to {msg['reply_to_user_name']})" if msg.get("reply_to_user_name") else ""
            lines.append(f"[ID: {msg.get('id')}] [{msg.get('timestamp_str')}] [{msg.get('from_user_name')}]{reply_part}: {msg.get('text')}")
        return "\n".join(lines)

    def _combined_store_memories(self) -> Dict[str, List[Dict[str, Any]]]:
        """Hot + deep entries concatenated, so forget analysis can target archived items too."""
        hot = self.memory.get_all_memories()
        deep = self.deep_memory.get_all_memories()
        return {
            "memories": list(hot.get("memories", [])) + list(deep.get("memories", [])),
            "dynamics": list(hot.get("dynamics", [])) + list(deep.get("dynamics", [])),
            "inside_jokes": list(hot.get("inside_jokes", [])) + list(deep.get("inside_jokes", [])),
        }

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

        # 3. Message entities (Telegram mentions in text or photo caption)
        all_entities = list(getattr(message, "entities", None) or []) + list(getattr(message, "caption_entities", None) or [])
        if all_entities and text:
            for entity in all_entities:
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
            uname_clean = bot_username.lstrip("@").strip()
            if uname_clean.lower().endswith("_bot") and len(uname_clean) > 4:
                candidates.append(uname_clean[:-4])
            elif uname_clean.lower().endswith("bot") and len(uname_clean) > 3:
                candidates.append(uname_clean[:-3])

        if bot_name and isinstance(bot_name, str):
            candidates.append(bot_name)
            for w in bot_name.split():
                clean_w = w.strip()
                if len(clean_w) >= 3 and clean_w.lower() not in ("bot", "the"):
                    candidates.append(clean_w)

        for default_name in ("Pletykás", "Pletykas", "Pletyi", "Pletyo", "Pletyó"):
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

        # 5. Forget request directed at the bot
        if self._is_forget_request(message, bot_username=bot_username, bot_name=bot_name, bot_id=bot_id):
            return True

        return False

    def _is_forget_request(
        self,
        message: Any,
        bot_username: str = "",
        bot_name: str = "",
        bot_id: Optional[int] = None,
    ) -> bool:
        """Determines if a message is asking the bot to forget something from memory."""
        text = ""
        if hasattr(message, "text") or hasattr(message, "caption"):
            text = getattr(message, "text", "") or getattr(message, "caption", "") or ""
        elif isinstance(message, dict):
            text = message.get("text", "") or message.get("caption", "") or ""
        elif isinstance(message, str):
            text = message

        if not text:
            return False

        clean_text = text.strip()
        lower_text = clean_text.lower()

        # Keywords indicating forget / delete / clear from memory intent
        forget_patterns = [
            r"\bfelejtsd\s+el\b",
            r"\bfelejts\s+el\b",
            r"\bfelejtse\s+el\b",
            r"\bfelejtsétek\s+el\b",
            r"\bfelejtsük\s+el\b",
            r"\bne\s+emlékezz(?:él|etek)?\b",
            r"\bforget\b",
            r"\berase\s+(?:.*?\s+)?(?:from\s+)?memory\b",
            r"\bclear\s+(?:.*?\s+)?(?:from\s+)?memory\b",
            r"\bdelete\s+(?:.*?\s+)?(?:from\s+)?memory\b",
            r"\bwipe\s+(?:.*?\s+)?(?:from\s+)?memory\b",
            r"\btöröld\s+(?:.*?\s+)?(?:a\s+memóriádból|az\s+emlékeidből|az\s+összes\s+emléket|a\s+memóriából)\b",
            r"\btorold\s+(?:.*?\s+)?(?:a\s+memoriadbol|az\s+emlekeidbol|az\s+osszes\s+emleket|a\s+memoriabol)\b",
        ]

        has_forget_keyword = any(re.search(pat, lower_text, re.IGNORECASE) for pat in forget_patterns)
        if not has_forget_keyword:
            return False

        # 1. If replying directly to bot's message
        reply_to = getattr(message, "reply_to_message", None) if not isinstance(message, dict) else None
        if reply_to and getattr(reply_to, "from_user", None):
            if reply_to.from_user.is_bot:
                return True

        # 2. Check candidate names for bot
        candidates: List[str] = []
        if bot_username:
            candidates.append(bot_username.lstrip("@"))
        if bot_name:
            candidates.append(bot_name)
        for default_name in ("Pletykás", "Pletykas", "Pletyi", "Pletyo", "Pletyó", "bot"):
            candidates.append(default_name)
        candidates.extend(self.state.get_nicknames())

        for cand in candidates:
            if cand and isinstance(cand, str):
                clean_cand = cand.strip().lstrip("@")
                if clean_cand and re.search(rf"(?:^|[^\w@])@?{re.escape(clean_cand.lower())}(?:[^\w]|$)", lower_text):
                    return True

        # 3. Explicit reference to bot's memory
        memory_refs = [
            r"memóriádból", r"memóriádban", r"emlékeidből", r"memóriából",
            r"memoriadbol", r"memoriadban", r"emlekeidbol", r"memoriabol",
            r"from\s+your\s+memory", r"in\s+your\s+memory", r"your\s+memory",
        ]
        if any(re.search(ref, lower_text) for ref in memory_refs):
            return True

        # 4. Starts directly with an imperative forget command
        direct_imperatives = [
            r"^(?:kérlek\s+|kerlek\s+|please\s+)?felejtsd\s+el\b",
            r"^(?:kérlek\s+|kerlek\s+|please\s+)?felejts\s+el\b",
            r"^(?:kérlek\s+|kerlek\s+|please\s+)?forget\b",
            r"^(?:kérlek\s+|kerlek\s+|please\s+)?töröld\s+(?:ki\s+)?a\s+memóriádból\b",
            r"^(?:kérlek\s+|kerlek\s+|please\s+)?torold\s+(?:ki\s+)?a\s+memoriadbol\b",
            r"^(?:kérlek\s+|kerlek\s+|please\s+)?töröld\s+az\s+összes\s+emléket\b",
            r"^(?:kérlek\s+|kerlek\s+|please\s+)?clear\s+(?:all\s+)?memory\b",
        ]
        if any(re.search(imp, clean_text, re.IGNORECASE) for imp in direct_imperatives):
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
            schedule_spec: Optional[Dict[str, Any]] = None

            if photo_entry:
                # Multimodal evaluation via Vision Model
                photo_bytes = base64.b64decode(photo_entry["media_b64"])
                user_caption = trigger_entry.get("text", "").replace("[Photo]", "").strip() if trigger_entry else ""
                reply_text, reaction, image_spec, img_desc, schedule_spec = await self.llm.describe_and_reply_image(
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
                reply_text, reaction, image_spec, schedule_spec = await self.llm.evaluate_and_reply(
                    system_prompt=sys_prompt,
                    memory_context=memory_ctx,
                    transcript=transcript,
                    bot_username=bot_username,
                    is_direct_trigger=is_direct_trigger,
                    talkativeness=talkativeness,
                )

            # Handle Schedule Actions (Create or Cancel)
            if schedule_spec:
                action = schedule_spec.get("action")
                if action == "create":
                    sched_type = schedule_spec.get("type", "oneshot")
                    time_str = schedule_spec.get("time") or schedule_spec.get("start") or ""
                    interval_str = schedule_spec.get("interval") if sched_type == "periodic" else None
                    interval_spec = None
                    interval_rd = None
                    if sched_type == "periodic":
                        interval_rd = parse_interval_relativedelta(interval_str or "", min_seconds=60)
                        if interval_rd is None:
                            interval_rd = relativedelta(days=1)
                            interval_str = "1 day"
                        interval_spec = serialize_interval(interval_rd)

                    now = self.state.get_current_time()
                    tz = self.state.get_tzinfo()
                    target_dt = None
                    if sched_type == "periodic" and not time_str:
                        target_dt = now + (interval_rd or relativedelta(days=1))
                    elif time_str:
                        target_dt = parse_schedule_time(time_str, tz, now)

                    if target_dt is None:
                        logger.warning("Unparseable schedule time '%s', defaulting to 60s from now", time_str)
                        target_dt = now + relativedelta(seconds=60)
                    elif target_dt <= now:
                        if sched_type == "periodic" and interval_rd:
                            while target_dt <= now:
                                target_dt += interval_rd
                        else:
                            target_dt = now + relativedelta(seconds=60)

                    sched_entry = {
                        "id": f"sched_{int(time.time())}_{trigger_msg_id or random.randint(1000, 9999)}",
                        "type": sched_type,
                        "target_msg_id": trigger_msg_id,
                        "target_time": target_dt.isoformat(),
                        "interval_str": interval_str,
                        "interval_spec": interval_spec,
                        "description": schedule_spec.get("description", "").strip(),
                        "created_at": now.isoformat(),
                    }
                    self.state.add_scheduled_reply(sched_entry)
                    self.schedule_reply_job(context, sched_entry)
                    logger.info("Registered and scheduled reply job: %s (type=%s, target=%s)", sched_entry["id"], sched_type, sched_entry["target_time"])

                elif action == "cancel":
                    sched_id = schedule_spec.get("schedule_id", "").strip()
                    if sched_id:
                        self.cancel_reply_job(context, sched_id)
                        removed = self.state.remove_scheduled_reply(sched_id)
                        logger.info("Cancelled scheduled reply '%s' (removed from state: %s)", sched_id, removed)
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

            # 2.5. Memory Clearing & Forgetting Action
            forget_spec = None
            if reply_text:
                reply_text, forget_spec = self.llm.extract_forget(reply_text)

            trigger_text = trigger_entry.get("text", "") if trigger_entry else ""
            sender_name = trigger_entry.get("from_user_name", "") if trigger_entry else ""
            bot_first = getattr(context.bot, "first_name", None) if context.bot else None
            bot_display_name = bot_first if isinstance(bot_first, str) else "Pletykas"
            bot_user = getattr(context.bot, "username", None) if context.bot else None
            bot_uname = bot_user if isinstance(bot_user, str) else ""
            is_forget_msg = self._is_forget_request(trigger_text, bot_username=bot_uname, bot_name=bot_display_name)

            if is_forget_msg:
                # The conversational model only sees hot entries, so an inline
                # <FORGET> spec cannot target archived (deep) topics. On an
                # explicit forget request the combined-store analysis is
                # authoritative and supersedes the inline spec; None (analysis
                # failure) keeps whatever the inline block proposed.
                logger.info("Forget request detected; running curate_forget analysis over combined hot+deep stores...")
                try:
                    analyzed_spec = await self.llm.curate_forget(
                        current_memories=self._combined_store_memories(),
                        forget_request_text=trigger_text,
                        sender_name=sender_name,
                        recent_transcript=transcript,
                    )
                except Exception as e:
                    logger.error("curate_forget failed: %s", e)
                    analyzed_spec = None
                if analyzed_spec is not None:
                    if forget_spec is not None:
                        logger.info("Replacing inline <FORGET> spec with curate_forget analysis.")
                    forget_spec = analyzed_spec

            if forget_spec:
                self.memory.create_backup()
                self.deep_memory.create_backup()
                # Apply the SAME spec to BOTH stores and sum the counters, so
                # archived entries are forgotten too (clear_all wipes both tiers).
                # apply_forget itself stays single-store; the caller loops.
                summary = self.memory.apply_forget(forget_spec)
                deep_summary = self.deep_memory.apply_forget(forget_spec)
                for key in ("discarded_facts", "updated_facts", "discarded_dynamics", "discarded_jokes"):
                    summary[key] = summary.get(key, 0) + deep_summary.get(key, 0)

                scrub_targets: List[str] = []
                if forget_spec.get("clear_all"):
                    self.state.clear_memory_history()
                else:
                    for f in forget_spec.get("facts_to_discard", []):
                        if isinstance(f, str) and f.strip():
                            scrub_targets.append(f.strip())
                    for u in forget_spec.get("facts_to_update", []):
                        if isinstance(u, dict) and u.get("topic"):
                            scrub_targets.append(str(u["topic"]).strip())
                    for d in forget_spec.get("dynamics_to_discard", []):
                        if isinstance(d, str) and d.strip():
                            scrub_targets.append(d.strip())
                    for j in forget_spec.get("jokes_to_discard", []):
                        if isinstance(j, str) and j.strip():
                            scrub_targets.append(j.strip())
                    for r in forget_spec.get("raw_targets", []):
                        if isinstance(r, str) and r.strip():
                            scrub_targets.append(r.strip())

                    if trigger_text:
                        m_cmd = re.search(
                            r"\b(?:felejtsd\s+el|felejts\s+el|felejtse\s+el|forget|töröld\s+(?:ki\s+)?(?:a\s+memóriádból)?)\s+(?:hogy\s+|that\s+|about\s+)?(.*)",
                            trigger_text,
                            re.IGNORECASE,
                        )
                        if m_cmd:
                            raw_target = m_cmd.group(1).strip()
                            clean_target = re.sub(r"[!?,.]+$", "", raw_target).strip()
                            if clean_target and len(clean_target) >= 3:
                                scrub_targets.append(clean_target)
                                for w in re.findall(r"\w+", clean_target):
                                    if len(w) >= 4 and w.lower() not in ("hogy", "that", "this", "about", "please", "kérlek", "kerlek"):
                                        scrub_targets.append(w)
                                        if len(w) >= 5 and w.endswith("t"):
                                            scrub_targets.append(w[:-1])

                    if scrub_targets:
                        scrubbed = self.state.scrub_memory_history(scrub_targets)
                        logger.info("Scrubbed %d messages from memory history matching forget targets", scrubbed)
                logger.info(
                    "Thoroughly cleared memory upon request: discarded_facts=%d, updated_facts=%d, discarded_dyn=%d, discarded_jokes=%d",
                    summary.get("discarded_facts", 0),
                    summary.get("updated_facts", 0),
                    summary.get("discarded_dynamics", 0),
                    summary.get("discarded_jokes", 0),
                )

                if self.state.is_debug_mode():
                    self._log_debug_group_msg(
                        "MEMORY_FORGET",
                        f"Forget Request: {trigger_text}\nApplied Spec: {json.dumps(forget_spec, ensure_ascii=False)}\nSummary: {json.dumps(summary)}",
                    )

                if not reply_text or reply_text.strip() == "<NO_REPLY>":
                    lang = self.state.get_language().lower()
                    if "hungarian" in lang or "magyar" in lang:
                        reply_text = "Rendben, ezt teljesen kitöröltem az emlékezetemből! 🤐"
                    else:
                        reply_text = "Understood, that has been thoroughly cleared from my memory! 🤐"

            # 3. Text Message and/or Poll Action
            poll_spec = None
            if reply_text:
                reply_text, poll_spec = self.llm.extract_poll(reply_text)

            if reply_text and self._suppress_repeated_reply(reply_text, context):
                reply_text = None

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

            # 4. Poll Action
            if poll_spec:
                poll_reply_to = trigger_msg_id if (is_direct_trigger and not reply_text) else None
                await self._send_group_poll(
                    context=context,
                    chat_id=chat_id,
                    question=poll_spec["question"],
                    options=poll_spec["options"],
                    reply_to_msg_id=poll_reply_to,
                    poll_type_label="Poll",
                )
        except Exception as e:
            logger.error("Error during evaluation execution: %s", e, exc_info=True)
            fallback_text = "sorry, got a bit tangled up in my thoughts... I'll be right back! 💫"
            if is_direct_trigger and not self._suppress_repeated_reply(fallback_text, context):
                try:
                    sent_err = await context.bot.send_message(
                        chat_id=chat_id,
                        text=fallback_text,
                        reply_to_message_id=trigger_msg_id,
                    )
                    if self.state.is_debug_mode():
                        self._log_debug_group_msg(
                            "OUTGOING",
                            f"Chat ID: {chat_id} | Message ID: {sent_err.message_id} | Reply to msg ID: {trigger_msg_id}\nType: Error Fallback Message\nText: {fallback_text}",
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
            self.trigger_curation_background()


        # Generate new spontaneous message random timestamp upon group message arrival
        if self.state.is_spontaneous_enabled():
            self.schedule_spontaneous_job(context, force_new=True)

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
        next_spont_dt = self.get_next_spontaneous_time(context)
        next_spont_str = self.state.format_time(next_spont_dt) if next_spont_dt else "None"
        spont_status = f"{'ON' if spont.get('enabled') else 'OFF'} (every {min_h:g}-{max_h:g}h, random) (Next: {next_spont_str})"
        grounding_status = "ON" if self.state.is_search_grounding_active() else "OFF"
        debug_status = "ON" if self.state.is_debug_mode() else "OFF"
        nicks = self.state.get_nicknames()
        nicks_str = ", ".join(nicks) if nicks else "None"
        tl_primary = self.params.model_thinking_level or "off (default)"
        tl_large = self.params.effective_large_thinking_level or "off (default)"
        tl_image = self.params.effective_image_thinking_level or "minimal (default)"

        # Scheduled replies stats
        sched_replies = self.state.get_scheduled_replies()
        sched_count = len(sched_replies)
        next_sched_str = "None"
        if sched_replies:
            valid_targets = []
            for r in sched_replies:
                try:
                    dt = datetime.fromisoformat(r["target_time"])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=self.state.get_tzinfo())
                    valid_targets.append(dt)
                except Exception:
                    pass
            if valid_targets:
                earliest = min(valid_targets)
                next_sched_str = self.state.format_time(earliest)
        sched_status = f"{sched_count} active (Next: {next_sched_str})"

        text = (
            f"📊 <b>Pletykas Status</b>\n"
            f"• Group Chat ID: <code>{self.params.group_chat_id}</code>\n"
            f"• Active Language: <code>{self.state.get_language()}</code>\n"
            f"• Timezone: <code>{self.state.get_timezone()}</code> ({self.state.get_current_time_str()})\n"
            f"• Talkativeness: <code>{self.state.get_talkativeness()}/10</code>\n"
            f"• Cooldown: <code>{self.state.get_cooldown_sec()}s</code>\n"
            f"• Nicknames: <code>{html.escape(nicks_str)}</code>\n"
            f"• Search Grounding: <code>{grounding_status}</code>\n"
            f"• Small Model Search: <code>{'ON (Direct Search)' if self.state.is_search_small_model() else 'OFF (Delegates to Large Model)'}</code>\n"
            f"• Image Vision Model: <code>{'Large Model (Instant)' if self.state.is_image_interpretation_large_model() else 'Small Model First'}</code>\n"
            f"• Debug Mode: <code>{debug_status}</code>\n"
            f"• Sleep Schedule: <code>{sleep_status}</code>\n"
            f"• Spontaneous Messages: <code>{spont_status}</code>\n"
            f"• Scheduled Replies: <code>{sched_status}</code>\n"
            f"• Models:\n"
            f"  - Primary: <code>{self.params.model_name}</code> (thinking: {tl_primary})\n"
            f"  - Larger: <code>{self.params.model_large_name}</code> (thinking: {tl_large})\n"
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
            next_spont_dt = self.get_next_spontaneous_time(context)
            next_spont_str = self.state.format_time(next_spont_dt) if next_spont_dt else "None"
            next_line = f"• Next Firing: <b>{next_spont_str}</b>\n" if spont.get("enabled") else ""
            await update.effective_message.reply_text(
                f"🎲 Spontaneous revival messages: <b>{status}</b>\n"
                f"• Interval: <b>{min_h:g} - {max_h:g} hours</b> (random)\n"
                f"{next_line}\n"
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
            self._next_spontaneous_time = None
            self.state.set_spontaneous_next_fire_time(None)
            await update.effective_message.reply_text("✅ Spontaneous revival messages turned <b>OFF</b>.", parse_mode=ParseMode.HTML)
        elif mode == "on":
            self.state.set_spontaneous_settings(enabled=True)
            self.state.set_spontaneous_next_fire_time(None)
            self.schedule_spontaneous_job(context)
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
            self.state.set_spontaneous_next_fire_time(None)
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
            if self.state.is_spontaneous_enabled():
                self.state.set_spontaneous_next_fire_time(None)
                self.schedule_spontaneous_job(context)
            await status_msg.edit_text(
                f"✅ <b>Spontaneous message dispatched to group:</b>\n\n{html.escape(detail)}",
                parse_mode=ParseMode.HTML,
            )
        else:
            await status_msg.edit_text(
                f"❌ Failed to dispatch spontaneous message: {html.escape(detail)}",
                parse_mode=ParseMode.HTML,
            )


    async def cmd_scheduled(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        subcommand = args[0].strip().lower() if args else "list"

        if subcommand in ("list", "show"):
            replies = self.state.get_scheduled_replies()
            if not replies:
                await update.effective_message.reply_text("📋 No active scheduled replies.")
                return

            lines = ["📋 <b>Active Scheduled Replies</b>:"]
            for r in replies:
                s_id = html.escape(str(r.get("id", "")))
                s_type = html.escape(str(r.get("type", "oneshot")))
                t_time = html.escape(str(r.get("target_time", "")))
                desc = html.escape(str(r.get("description", "")))
                if s_type == "periodic":
                    interval = html.escape(str(r.get("interval_str", "")))
                    lines.append(
                        f"\n• <b>ID:</b> <code>{s_id}</code>\n"
                        f"  <b>Type:</b> periodic (every {interval})\n"
                        f"  <b>Next:</b> <code>{t_time}</code>\n"
                        f"  <b>Task:</b> {desc}"
                    )
                else:
                    lines.append(
                        f"\n• <b>ID:</b> <code>{s_id}</code>\n"
                        f"  <b>Type:</b> oneshot\n"
                        f"  <b>Due:</b> <code>{t_time}</code>\n"
                        f"  <b>Task:</b> {desc}"
                    )
            lines.append("\n<i>To cancel a schedule:</i> <code>/scheduled cancel &lt;id&gt;</code>")
            await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            return

        if subcommand == "cancel":
            if len(args) < 2:
                await update.effective_message.reply_text("❌ Usage: <code>/scheduled cancel &lt;id&gt;</code>", parse_mode=ParseMode.HTML)
                return
            target_id = args[1].strip()
            self.cancel_reply_job(context, target_id)
            removed = self.state.remove_scheduled_reply(target_id)
            if removed:
                await update.effective_message.reply_text(f"✅ Scheduled reply <code>{html.escape(target_id)}</code> has been cancelled.", parse_mode=ParseMode.HTML)
            else:
                await update.effective_message.reply_text(f"❌ Scheduled reply <code>{html.escape(target_id)}</code> not found.", parse_mode=ParseMode.HTML)
            return

        await update.effective_message.reply_text(
            "❌ Usage: <code>/scheduled</code> (or <code>/scheduled list</code>) to view, or <code>/scheduled cancel &lt;id&gt;</code> to cancel.",
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

    async def cmd_search_small(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update.effective_user.id if update.effective_user else None):
            return
        if update.effective_chat and update.effective_chat.type != "private":
            return

        args = context.args or []
        if not args:
            cur = "ON" if self.state.is_search_small_model() else "OFF"
            desc = "Small model searches directly when search grounding is active." if cur == "ON" else "Small model delegates web searches to the large model (with automatic escalation)."
            await update.effective_message.reply_text(
                f"🔍 Small Model Web Search is currently: <b>{cur}</b>\n<i>({desc})</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        mode = args[0].lower()
        if mode == "on":
            self.state.set_search_small_model(True)
            await update.effective_message.reply_text(
                "✅ Small model web search turned <b>ON</b>. Small model will search directly if supported.",
                parse_mode=ParseMode.HTML,
            )
        elif mode == "off":
            self.state.set_search_small_model(False)
            await update.effective_message.reply_text(
                "✅ Small model web search turned <b>OFF</b>. Small model will delegate web searches to the large model.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.effective_message.reply_text(
                "❌ Usage: <code>/search_small [on|off]</code>",
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
        if not summary:
            await msg.edit_text("⚠️ Memory consolidation finished with no changes or was already in progress.")
            return
        text = (
            f"✅ <b>Memory Consolidation Complete:</b>\n"
            f"• Facts Added: {summary.get('added_facts', 0)}\n"
            f"• Facts Updated: {summary.get('updated_facts', 0)}\n"
            f"• Facts Discarded: {summary.get('discarded_facts', 0)}\n"
            f"• Dynamics Added: {summary.get('added_dynamics', 0)}\n"
            f"• Dynamics Discarded: {summary.get('discarded_dynamics', 0)}\n"
            f"• Jokes Added: {summary.get('added_jokes', 0)}\n"
            f"• Jokes Discarded: {summary.get('discarded_jokes', 0)}"
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
        application.add_handler(CommandHandler(["search_small", "search_small_model", "searchsmall", "grounding_small"], self.cmd_search_small))
        application.add_handler(CommandHandler("cancel", self.cmd_cancel))
        application.add_handler(CommandHandler("curate", self.cmd_curate))

        application.add_handler(CommandHandler(["scheduled", "schedules"], self.cmd_scheduled))
        # Main message handler (captures text, photos, media in groups and private)
        application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, self.on_message))
