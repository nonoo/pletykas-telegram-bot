import glob
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def validate_memory_dict(data: Any) -> tuple[bool, str, Optional[Dict[str, Any]]]:
    """Validates and sanitizes a memory JSON structure, stripping any legacy IDs."""
    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object/dict", None

    for key in ("memories", "dynamics", "inside_jokes"):
        if key in data and not isinstance(data[key], list):
            return False, f"Field \"{key}\" must be a list", None

    cleaned_memories = []
    for idx, item in enumerate(data.get("memories", [])):
        if not isinstance(item, dict):
            return False, f"Memory entry #{idx + 1} must be an object", None
        topic = item.get("topic")
        content = item.get("content")
        if not topic or not isinstance(topic, str) or not topic.strip():
            return False, f"Memory entry #{idx + 1} missing required string field \"topic\"", None
        if not content or not isinstance(content, str) or not content.strip():
            return False, f"Memory entry #{idx + 1} missing required string field \"content\"", None
        now_str = _utc_iso_now()
        cleaned_memories.append({
            "topic": topic.strip(),
            "content": content.strip(),
            "created_at": item.get("created_at") or now_str,
        })

    cleaned_dynamics = []
    for idx, item in enumerate(data.get("dynamics", [])):
        if not isinstance(item, dict):
            return False, f"Dynamic entry #{idx + 1} must be an object", None
        members = item.get("members")
        relation = item.get("relation")
        if not members or not isinstance(members, list) or not all(isinstance(m, str) and m.strip() for m in members):
            return False, f"Dynamic entry #{idx + 1} missing required non-empty list \"members\"", None
        if not relation or not isinstance(relation, str) or not relation.strip():
            return False, f"Dynamic entry #{idx + 1} missing required string field \"relation\"", None
        cleaned_dynamics.append({
            "members": [m.strip() for m in members],
            "relation": relation.strip(),
            "created_at": item.get("created_at") or _utc_iso_now(),
        })

    cleaned_jokes = []
    for idx, item in enumerate(data.get("inside_jokes", [])):
        if not isinstance(item, dict):
            return False, f"Inside joke entry #{idx + 1} must be an object", None
        title = item.get("title")
        context = item.get("context")
        if not title or not isinstance(title, str) or not title.strip():
            return False, f"Inside joke entry #{idx + 1} missing required string field \"title\"", None
        if not context or not isinstance(context, str) or not context.strip():
            return False, f"Inside joke entry #{idx + 1} missing required string field \"context\"", None
        cleaned_jokes.append({
            "title": title.strip(),
            "context": context.strip(),
            "created_at": item.get("created_at") or _utc_iso_now(),
        })

    cleaned = {
        "version": int(data.get("version", 1)),
        "last_curated_at": str(data.get("last_curated_at", "")),
        "memories": cleaned_memories,
        "dynamics": cleaned_dynamics,
        "inside_jokes": cleaned_jokes,
    }
    return True, "", cleaned



class MemoryManager:
    def __init__(self, file_path: str = "pletykas-memory.json"):
        self.file_path = os.path.abspath(file_path)
        self.data: Dict[str, Any] = self._create_default_memory()

    def _create_default_memory(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "last_curated_at": "",
            "memories": [],
            "dynamics": [],
            "inside_jokes": [],
        }

    def load(self) -> None:
        if not os.path.exists(self.file_path):
            self.data = self._create_default_memory()
            logger.info("Memory file %s does not exist. Initialized empty memory.", self.file_path)
            try:
                self.save()
            except Exception as e:
                logger.error("Failed to save initial memory to %s: %s", self.file_path, e)
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    defaults = self._create_default_memory()
                    for k, v in defaults.items():
                        if k not in loaded:
                            loaded[k] = v
                    if not isinstance(loaded.get("memories"), list):
                        loaded["memories"] = []
                    if not isinstance(loaded.get("dynamics"), list):
                        loaded["dynamics"] = []
                    if not isinstance(loaded.get("inside_jokes"), list):
                        loaded["inside_jokes"] = []
                    # Strip any legacy "id" and "updated_at" fields from loaded entries
                    for m in loaded["memories"]:
                        if isinstance(m, dict):
                            m.pop("id", None)
                            m.pop("updated_at", None)
                    for d in loaded["dynamics"]:
                        if isinstance(d, dict):
                            d.pop("id", None)
                    for j in loaded["inside_jokes"]:
                        if isinstance(j, dict):
                            j.pop("id", None)
                    self.data = loaded
                else:
                    self.data = self._create_default_memory()
            logger.info("Loaded memory from %s (facts: %d, dynamics: %d, jokes: %d)",
                        self.file_path,
                        len(self.data["memories"]),
                        len(self.data["dynamics"]),
                        len(self.data["inside_jokes"]))
        except Exception as e:
            logger.error("Failed to load memory file %s: %s", self.file_path, e)
            self.data = self._create_default_memory()

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
            logger.debug("Successfully saved memory to %s", self.file_path)
        except Exception as e:
            logger.error("Error saving memory to %s: %s", self.file_path, e)
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            raise

    def create_backup(self) -> str:
        if not os.path.exists(self.file_path):
            return ""

        dir_name = os.path.dirname(os.path.abspath(self.file_path)) or "."
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        backup_path = os.path.join(dir_name, f"{base_name}-{timestamp}.json.bak")

        try:
            shutil.copy2(self.file_path, backup_path)
            logger.info("Created memory backup: %s", backup_path)

            # Rotate backups: retain up to 20
            pattern = os.path.join(dir_name, f"{base_name}-*.json.bak")
            existing = sorted(glob.glob(pattern))
            if len(existing) > 20:
                to_delete = existing[:-20]
                for old_bak in to_delete:
                    try:
                        os.remove(old_bak)
                        logger.debug("Pruned old memory backup: %s", old_bak)
                    except OSError as e:
                        logger.warning("Failed to remove old backup %s: %s", old_bak, e)

            return backup_path
        except Exception as e:
            logger.error("Failed to create memory backup: %s", e)
            return ""

    # Facts / Memories
    def add_memory(self, topic: str, content: str) -> None:
        memories = self.data.setdefault("memories", [])
        now_str = _utc_iso_now()
        entry = {
            "topic": topic.strip(),
            "content": content.strip(),
            "created_at": now_str,
        }
        memories.append(entry)
        self.save()

    def update_memory(self, topic: str, content: str) -> bool:
        topic_clean = topic.strip().lower()
        for entry in self.data.get("memories", []):
            if entry.get("topic", "").strip().lower() == topic_clean:
                entry["content"] = content.strip()
                self.save()
                return True
        return False

    def discard_memory(self, target: str) -> bool:
        clean = target.strip().lower()
        if not clean:
            return False
        memories = self.data.get("memories", [])
        initial_len = len(memories)
        self.data["memories"] = [
            m for m in memories
            if clean != m.get("topic", "").strip().lower()
            and clean != m.get("content", "").strip().lower()
            and clean not in m.get("content", "").strip().lower()
            and clean not in m.get("topic", "").strip().lower()
        ]
        if len(self.data["memories"]) < initial_len:
            self.save()
            return True
        return False

    # Dynamics
    def add_dynamic(self, members: List[str], relation: str) -> None:
        dynamics = self.data.setdefault("dynamics", [])
        entry = {
            "members": [m.strip() for m in members if m.strip()],
            "relation": relation.strip(),
            "created_at": _utc_iso_now(),
        }
        dynamics.append(entry)
        self.save()

    def discard_dynamic(self, target: str) -> bool:
        clean = target.strip().lower()
        if not clean:
            return False
        dynamics = self.data.get("dynamics", [])
        initial_len = len(dynamics)
        self.data["dynamics"] = [
            d for d in dynamics
            if clean != d.get("relation", "").strip().lower()
            and clean not in d.get("relation", "").strip().lower()
            and not any(clean == m.strip().lower() for m in d.get("members", []))
        ]
        if len(self.data["dynamics"]) < initial_len:
            self.save()
            return True
        return False

    # Inside Jokes
    def add_inside_joke(self, title: str, context: str) -> None:
        jokes = self.data.setdefault("inside_jokes", [])
        entry = {
            "title": title.strip(),
            "context": context.strip(),
            "created_at": _utc_iso_now(),
        }
        jokes.append(entry)
        self.save()

    def discard_inside_joke(self, target: str) -> bool:
        clean = target.strip().lower()
        if not clean:
            return False
        jokes = self.data.get("inside_jokes", [])
        initial_len = len(jokes)
        self.data["inside_jokes"] = [
            j for j in jokes
            if clean != j.get("title", "").strip().lower()
            and clean != j.get("context", "").strip().lower()
            and clean not in j.get("context", "").strip().lower()
            and clean not in j.get("title", "").strip().lower()
        ]
        if len(self.data["inside_jokes"]) < initial_len:
            self.save()
            return True
        return False

    # Generic Discard
    def discard_any(self, target: str) -> bool:
        clean = target.strip().lower()
        return (
            self.discard_memory(clean)
            or self.discard_dynamic(clean)
            or self.discard_inside_joke(clean)
        )

    def clear(self) -> None:
        self.data["memories"] = []
        self.data["dynamics"] = []
        self.data["inside_jokes"] = []
        self.save()

    def get_all_memories(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "memories": list(self.data.get("memories", [])),
            "dynamics": list(self.data.get("dynamics", [])),
            "inside_jokes": list(self.data.get("inside_jokes", [])),
        }

    def set_last_curated_at(self, timestamp_iso: str) -> None:
        self.data["last_curated_at"] = timestamp_iso
        self.save()

    def get_last_curated_at(self) -> str:
        return self.data.get("last_curated_at", "")

    def format_for_context(self) -> str:
        memories = self.data.get("memories", [])
        dynamics = self.data.get("dynamics", [])
        jokes = self.data.get("inside_jokes", [])

        if not memories and not dynamics and not jokes:
            return "<memory>None</memory>"

        sections = ["<memory>"]

        if memories:
            sections.append("[Facts]")
            for m in memories:
                sections.append(f"- ({m.get('topic')}): {m.get('content')}")
            sections.append("")

        if dynamics:
            sections.append("[Group Dynamics]")
            for d in dynamics:
                members_str = " & ".join(d.get("members", []))
                sections.append(f"- ({members_str}): {d.get('relation')}")
            sections.append("")

        if jokes:
            sections.append("[Inside Jokes & Lore]")
            for j in jokes:
                sections.append(f"- ({j.get('title')}): {j.get('context')}")
            sections.append("")

        # Strip any trailing blank line inside the tags
        if sections[-1] == "":
            sections.pop()

        sections.append("</memory>")
        return "\n".join(sections)
