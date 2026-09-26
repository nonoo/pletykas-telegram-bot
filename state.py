import json
import logging
import os
import tempfile
import time as time_mod
from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import zoneinfo

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Pletykas, an observant, witty, and subtly gossipy member of this Telegram group. "
    "You actively follow the group's dynamics, personal stories, inside jokes, and daily banter. "
    "You participate naturally as a friend, not an assistant. You have genuine opinions, a playful sense of humor, "
    "and never sound like a corporate AI."
)

CHAT_HISTORY_SIZE = 20
MEMORY_HISTORY_SIZE = 30


class StateManager:
    def __init__(self, file_path: str = "pletykas-state.json", group_chat_id: int = 0):
        self.file_path = os.path.abspath(file_path)
        self.initial_group_chat_id = group_chat_id
        self.data: Dict[str, Any] = self._create_default_state()

    def _create_default_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "group_chat_id": self.initial_group_chat_id,
            "language": "English",
            "timezone": "UTC",
            "talkativeness": 5,
            "cooldown_sec": 5,
            "search_grounding": True,
            "search_small_model": False,
            "image_interpretation_large_model": True,
            "debug": False,
            "nicknames": ["pletyi", "pletyo"],
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "sleep_schedule": {
                "enabled": True,
                "sleep_start": "23:00",
                "sleep_end": "07:00",
            },
            "spontaneous_messages": {
                "enabled": True,
                "min_hours": 2.0,
                "max_hours": 4.0,
                "next_fire_time": None,
            },
            "chat_history": [],
            "memory_history": [],
            "messages_since_last_curation": 0,
            "scheduled_replies": [],
        }

    def load(self) -> None:
        if not os.path.exists(self.file_path):
            self.data = self._create_default_state()
            logger.info("State file %s does not exist. Initialized default state.", self.file_path)
            try:
                self.save()
            except Exception as e:
                logger.error("Failed to save initial default state to %s: %s", self.file_path, e)
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    defaults = self._create_default_state()
                    # Ensure all default keys exist
                    for k, v in defaults.items():
                        if k not in loaded:
                            loaded[k] = v
                    loaded.pop("llm_usage", None)
                    loaded.pop("stats", None)
                    spont = loaded.get("spontaneous_messages")
                    if not isinstance(spont, dict):
                        loaded["spontaneous_messages"] = {
                            "enabled": True,
                            "min_hours": 2.0,
                            "max_hours": 4.0,
                            "next_fire_time": None,
                        }
                    else:
                        if "enabled" not in spont:
                            spont["enabled"] = True
                        if "min_hours" not in spont:
                            spont["min_hours"] = 2.0
                        if "max_hours" not in spont:
                            spont["max_hours"] = 4.0
                        if "next_fire_time" not in spont:
                            spont["next_fire_time"] = None
                    if not isinstance(loaded.get("chat_history"), list):
                        loaded["chat_history"] = []
                    if not isinstance(loaded.get("memory_history"), list):
                        loaded["memory_history"] = []
                    if not isinstance(loaded.get("scheduled_replies"), list):
                        loaded["scheduled_replies"] = []
                    self.data = loaded
                else:
                    self.data = self._create_default_state()
            logger.info("Loaded state from %s (group_chat_id: %s)", self.file_path, self.data.get("group_chat_id"))
        except Exception as e:
            logger.error("Failed to load state file %s: %s", self.file_path, e)
            self.data = self._create_default_state()

    def reload(self) -> None:
        self.load()

    def save(self) -> None:
        dir_name = os.path.dirname(os.path.abspath(self.file_path)) or "."
        os.makedirs(dir_name, exist_ok=True)

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                temp_file = tf.name
                json.dump(self.data, tf, indent=2, ensure_ascii=False)
            os.replace(temp_file, self.file_path)
            logger.debug("Successfully saved state to %s", self.file_path)
        except Exception as e:
            logger.error("Error saving state to %s: %s", self.file_path, e)
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            raise

    # Timezone & Time utilities
    def get_timezone(self) -> str:
        return self.data.get("timezone", "UTC")

    def set_timezone(self, tz_name: str) -> bool:
        tz_clean = tz_name.strip()
        try:
            zoneinfo.ZoneInfo(tz_clean)
            self.data["timezone"] = tz_clean
            self.save()
            return True
        except Exception:
            return False

    def get_tzinfo(self) -> zoneinfo.ZoneInfo:
        tz_name = self.get_timezone()
        try:
            return zoneinfo.ZoneInfo(tz_name)
        except Exception:
            return zoneinfo.ZoneInfo("UTC")

    def get_current_time(self) -> datetime:
        return datetime.now(self.get_tzinfo())

    def get_current_time_str(self) -> str:
        return self.get_current_time().strftime("%Y-%m-%d %H:%M:%S")

    def format_time(self, dt_or_epoch: Union[datetime, float, int]) -> str:
        tz = self.get_tzinfo()
        if isinstance(dt_or_epoch, (int, float)):
            dt = datetime.fromtimestamp(dt_or_epoch, tz=timezone.utc).astimezone(tz)
        elif isinstance(dt_or_epoch, datetime):
            if dt_or_epoch.tzinfo is None:
                dt = dt_or_epoch.replace(tzinfo=timezone.utc).astimezone(tz)
            else:
                dt = dt_or_epoch.astimezone(tz)
        else:
            return self.get_current_time_str()
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    # Group Chat ID
    def get_group_chat_id(self) -> int:
        return int(self.data.get("group_chat_id", self.initial_group_chat_id))

    def set_group_chat_id(self, chat_id: int) -> None:
        self.data["group_chat_id"] = chat_id
        self.save()

    # Language
    def get_language(self) -> str:
        return self.data.get("language", "English")

    def set_language(self, lang: str) -> None:
        self.data["language"] = lang.strip()
        self.save()

    # System prompt
    def get_system_prompt(self) -> str:
        return self.data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    def set_system_prompt(self, prompt: str) -> None:
        self.data["system_prompt"] = prompt.strip()
        self.save()

    def reset_system_prompt(self) -> None:
        self.data["system_prompt"] = DEFAULT_SYSTEM_PROMPT
        self.save()

    def get_effective_system_prompt(self) -> str:
        base = self.get_system_prompt()
        lang = self.get_language()
        prompt = (
            f"{base}\n\n"
            f"Always communicate in {lang}.\n"
            "Use Telegram HTML markdown. Only use the following HTML tags b, i, u, s, a, code, blockquote. Do not use any other HTML tags. Do not use LaTeX for formatting."
        )
        nicks = self.get_nicknames()
        if nicks:
            prompt += f"\nYour nicknames: {', '.join(nicks)}."
        return prompt

    # Talkativeness & Cooldown
    def get_talkativeness(self) -> int:
        return int(self.data.get("talkativeness", 5))

    def set_talkativeness(self, level: int) -> None:
        clamped = max(1, min(10, int(level)))
        self.data["talkativeness"] = clamped
        self.save()

    def get_cooldown_sec(self) -> int:
        return int(self.data.get("cooldown_sec", 5))

    def set_cooldown_sec(self, sec: int) -> None:
        clamped = max(0, int(sec))
        self.data["cooldown_sec"] = clamped
        self.save()

    # Search Grounding
    def is_search_grounding_active(self) -> bool:
        return bool(self.data.get("search_grounding", False))

    def set_search_grounding_active(self, active: bool) -> None:
        self.data["search_grounding"] = bool(active)
        self.save()

    # Small Model Search Selection
    def is_search_small_model(self) -> bool:
        return bool(self.data.get("search_small_model", False))

    def set_search_small_model(self, active: bool) -> None:
        self.data["search_small_model"] = bool(active)
        self.save()

    # Image Interpretation Model Selection
    def is_image_interpretation_large_model(self) -> bool:
        return bool(self.data.get("image_interpretation_large_model", True))

    def set_image_interpretation_large_model(self, active: bool) -> None:
        self.data["image_interpretation_large_model"] = bool(active)
        self.save()

    # Sleep Schedule
    def get_sleep_schedule(self) -> Dict[str, Any]:
        return self.data.get("sleep_schedule", {"enabled": False, "sleep_start": "23:00", "sleep_end": "07:00"})

    def set_sleep_schedule(self, enabled: bool, sleep_start: Optional[str] = None, sleep_end: Optional[str] = None) -> None:
        sched = self.get_sleep_schedule().copy()
        sched["enabled"] = bool(enabled)
        if sleep_start:
            sched["sleep_start"] = sleep_start.strip()
        if sleep_end:
            sched["sleep_end"] = sleep_end.strip()
        self.data["sleep_schedule"] = sched
        self.save()

    def is_sleeping(self) -> bool:
        sched = self.get_sleep_schedule()
        if not sched.get("enabled", False):
            return False

        try:
            start_parts = [int(p) for p in sched.get("sleep_start", "23:00").split(":")]
            end_parts = [int(p) for p in sched.get("sleep_end", "07:00").split(":")]
            t_start = time(start_parts[0], start_parts[1])
            t_end = time(end_parts[0], end_parts[1])

            now_time = self.get_current_time().time()

            if t_start <= t_end:
                # Same day sleep interval (e.g. 01:00 to 06:00)
                return t_start <= now_time < t_end
            else:
                # Overnight sleep interval (e.g. 23:00 to 07:00)
                return now_time >= t_start or now_time < t_end
        except Exception as e:
            logger.warning("Error evaluating sleep schedule: %s", e)
            return False

    # Spontaneous Messages
    def get_spontaneous_settings(self) -> Dict[str, Any]:
        spont = self.data.get("spontaneous_messages")
        if not isinstance(spont, dict):
            return {"enabled": False, "min_hours": 2.0, "max_hours": 4.0, "next_fire_time": None}
        return {
            "enabled": bool(spont.get("enabled", False)),
            "min_hours": float(spont.get("min_hours", 2.0)),
            "max_hours": float(spont.get("max_hours", 4.0)),
            "next_fire_time": spont.get("next_fire_time"),
        }

    def is_spontaneous_enabled(self) -> bool:
        return bool(self.get_spontaneous_settings().get("enabled", False))

    def set_spontaneous_settings(
        self,
        enabled: Optional[bool] = None,
        min_hours: Optional[float] = None,
        max_hours: Optional[float] = None,
        next_fire_time: Optional[Any] = None,
    ) -> None:
        spont = self.data.setdefault("spontaneous_messages", {})
        if enabled is not None:
            spont["enabled"] = bool(enabled)
        if min_hours is not None:
            spont["min_hours"] = float(min_hours)
        if max_hours is not None:
            spont["max_hours"] = float(max_hours)
        if next_fire_time is not None:
            spont["next_fire_time"] = next_fire_time
        self.save()

    def get_spontaneous_next_fire_time(self) -> Optional[datetime]:
        """Returns the persisted next spontaneous message fire time as a localized datetime, or None."""
        spont = self.data.get("spontaneous_messages")
        if not isinstance(spont, dict):
            return None
        val = spont.get("next_fire_time")
        if not val:
            return None
        tz = self.get_tzinfo()
        if isinstance(val, (int, float)):
            try:
                return datetime.fromtimestamp(val, tz=timezone.utc).astimezone(tz)
            except Exception:
                return None
        if isinstance(val, str):
            try:
                dt = datetime.fromisoformat(val)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=tz)
                return dt.astimezone(tz)
            except Exception:
                return None
        return None

    def set_spontaneous_next_fire_time(self, dt: Optional[datetime]) -> None:
        """Persists the next spontaneous message fire time into state."""
        spont = self.data.setdefault("spontaneous_messages", {})
        if dt is None:
            spont["next_fire_time"] = None
        else:
            tz = self.get_tzinfo()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            else:
                dt = dt.astimezone(tz)
            spont["next_fire_time"] = dt.isoformat()
        self.save()

    def set_spontaneous_interval(self, min_hours: float, max_hours: float) -> None:
        spont = self.data.setdefault("spontaneous_messages", {})
        spont["min_hours"] = float(min_hours)
        spont["max_hours"] = float(max_hours)
        self.save()
    # Debug Mode
    def is_debug_mode(self) -> bool:
        return bool(self.data.get("debug", False))

    def set_debug_mode(self, enabled: bool) -> None:
        self.data["debug"] = bool(enabled)
        self.save()

    # Nicknames
    def get_nicknames(self) -> List[str]:
        return list(self.data.get("nicknames", []))

    def set_nicknames(self, nicknames: List[str]) -> None:
        seen = set()
        clean_list: List[str] = []
        for n in nicknames:
            clean = str(n).strip().lstrip("@").strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                clean_list.append(clean)
        self.data["nicknames"] = clean_list
        self.save()

    # Chat History
    def get_chat_history(self) -> List[Dict[str, Any]]:
        return list(self.data.get("chat_history", []))

    def set_chat_history(self, history: List[Dict[str, Any]]) -> None:
        self.data["chat_history"] = list(history[-CHAT_HISTORY_SIZE:])
        self.save()

    def append_chat_message(self, msg: Dict[str, Any]) -> None:
        history = self.data.setdefault("chat_history", [])
        history.append(msg)
        if len(history) > CHAT_HISTORY_SIZE:
            self.data["chat_history"] = history[-CHAT_HISTORY_SIZE:]
        self.save()

    # Memory History (for curation)
    def get_memory_history(self) -> List[Dict[str, Any]]:
        return list(self.data.get("memory_history", []))

    def set_memory_history(self, history: List[Dict[str, Any]]) -> None:
        self.data["memory_history"] = list(history[-MEMORY_HISTORY_SIZE:])
        self.save()

    def append_memory_message(self, msg: Dict[str, Any]) -> None:
        history = self.data.setdefault("memory_history", [])
        history.append(msg)
        if len(history) > MEMORY_HISTORY_SIZE:
            self.data["memory_history"] = history[-MEMORY_HISTORY_SIZE:]
        self.save()

    # Curation counter
    def get_messages_since_last_curation(self) -> int:
        return int(self.data.get("messages_since_last_curation", 0))

    def set_messages_since_last_curation(self, count: int) -> None:
        self.data["messages_since_last_curation"] = max(0, int(count))
        self.save()

    def increment_messages_since_last_curation(self) -> int:
        cnt = self.get_messages_since_last_curation() + 1
        self.data["messages_since_last_curation"] = cnt
        self.save()
        return cnt

    # Stats
    def get_stats(self) -> Dict[str, Any]:
        return {
            "uncurated_messages": self.get_messages_since_last_curation(),
        }

    # Scheduled Replies
    def get_scheduled_replies(self) -> List[Dict[str, Any]]:
        return [dict(x) for x in self.data.get("scheduled_replies", [])]

    def get_scheduled_reply(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        for r in self.data.get("scheduled_replies", []):
            if r.get("id") == schedule_id:
                return dict(r)
        return None

    def add_scheduled_reply(self, entry: Dict[str, Any]) -> str:
        replies = self.data.setdefault("scheduled_replies", [])
        entry_copy = dict(entry)
        if not entry_copy.get("id"):
            entry_copy["id"] = f"sched_{int(time_mod.time())}_{os.urandom(2).hex()}"
        replies.append(entry_copy)
        self.save()
        return entry_copy["id"]

    def remove_scheduled_reply(self, schedule_id: str) -> bool:
        replies = self.data.get("scheduled_replies", [])
        initial_len = len(replies)
        self.data["scheduled_replies"] = [r for r in replies if r.get("id") != schedule_id]
        if len(self.data["scheduled_replies"]) != initial_len:
            self.save()
            return True
        return False

    def update_scheduled_reply(self, schedule_id: str, updates: Dict[str, Any]) -> bool:
        replies = self.data.get("scheduled_replies", [])
        for r in replies:
            if r.get("id") == schedule_id:
                r.update(updates)
                self.save()
                return True
        return False

    def format_scheduled_replies_context(self) -> str:
        replies = self.get_scheduled_replies()
        if not replies:
            return "None"
        lines = []
        for r in replies:
            s_id = r.get("id", "")
            s_type = r.get("type", "oneshot")
            t_time = r.get("target_time", "")
            desc = r.get("description", "")
            if s_type == "periodic":
                interval = r.get("interval_str", "")
                lines.append(f"- ID: {s_id} | Type: periodic | Next: {t_time} | Interval: {interval} | Description: {desc}")
            else:
                lines.append(f"- ID: {s_id} | Type: oneshot | Due: {t_time} | Description: {desc}")
        return "\n".join(lines)
