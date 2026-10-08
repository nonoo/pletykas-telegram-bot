import glob
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Tuple, Union

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
        created_at = item.get("created_at") or now_str
        updated_at = item.get("updated_at")
        if not isinstance(updated_at, str) or not updated_at.strip():
            updated_at = created_at
        cleaned_memories.append({
            "topic": topic.strip(),
            "content": content.strip(),
            "created_at": created_at,
            "updated_at": updated_at,
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
        self._lock = threading.RLock()
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
        with self._lock:
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
                        # Strip legacy "id" fields; keep "updated_at" (resets the archive-age clock)
                        for m in loaded["memories"]:
                            if isinstance(m, dict):
                                m.pop("id", None)
                                if not m.get("updated_at"):
                                    m["updated_at"] = m.get("created_at", "")
                        for d in loaded["dynamics"]:
                            if isinstance(d, dict):
                                d.pop("id", None)
                        for j in loaded["inside_jokes"]:
                            if isinstance(j, dict):
                                j.pop("id", None)
                        self.data = loaded
                    else:
                        logger.critical("Fatal: Memory file %s does not contain a JSON object. Quitting.", self.file_path)
                        sys.exit(1)
                logger.info("Loaded memory from %s (facts: %d, dynamics: %d, jokes: %d)",
                            self.file_path,
                            len(self.data["memories"]),
                            len(self.data["dynamics"]),
                            len(self.data["inside_jokes"]))
            except Exception as e:
                logger.critical("Fatal: Failed to load or parse memory file %s: %s. Quitting.", self.file_path, e)
                sys.exit(1)
    def reload(self) -> None:
        with self._lock:
            self.load()

    def save(self) -> None:
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            memories = self.data.setdefault("memories", [])
            now_str = _utc_iso_now()
            entry = {
                "topic": topic.strip(),
                "content": content.strip(),
                "created_at": now_str,
                "updated_at": now_str,
            }
            memories.append(entry)
            self.save()

    def update_memory(self, topic: str, content: str) -> bool:
        with self._lock:
            topic_clean = topic.strip().lower()
            for entry in self.data.get("memories", []):
                if entry.get("topic", "").strip().lower() == topic_clean:
                    entry["content"] = content.strip()
                    entry["updated_at"] = _utc_iso_now()
                    self.save()
                    return True
            return False

    def _matches_memory_target(self, m: Dict[str, Any], target: Any) -> bool:
        m_topic = str(m.get("topic", "")).strip().lower()
        m_content = str(m.get("content", "")).strip().lower()
        if not m_topic and not m_content:
            return False

        if isinstance(target, dict):
            t_topic = str(target.get("topic", "")).strip().lower()
            t_content = str(target.get("content", "")).strip().lower()
            if t_topic and t_topic == m_topic:
                if not t_content or t_content in m_content or m_content in t_content:
                    return True
            if t_content and (t_content == m_content or t_content in m_content or m_content in t_content):
                return True
            return False

        clean = str(target).strip().lower()
        if not clean:
            return False

        # Exact or substring match in topic or content
        if clean == m_topic or clean == m_content:
            return True
        if clean in m_content or clean in m_topic:
            return True

        # Parsed "topic: content" or "(topic): content" or "- (topic): content"
        prefix_match = re.match(r"^(?:-\s*)?(?:\(([^)]+)\)|([^:]+)):\s*(.*)$", clean)
        if prefix_match:
            cand_topic = (prefix_match.group(1) or prefix_match.group(2) or "").strip().lower()
            cand_content = prefix_match.group(3).strip().lower()
            if cand_topic and (cand_topic == m_topic or cand_topic in m_topic or m_topic in cand_topic):
                if not cand_content or cand_content in m_content or m_content in cand_content:
                    return True

        # If topic is in target (e.g. "Alice lives in Berlin" where topic is "Alice")
        if m_topic and m_topic in clean:
            content_words = [w for w in re.findall(r"\w+", m_content) if len(w) >= 3]
            clean_words = [w for w in re.findall(r"\w+", clean) if len(w) >= 3]
            if not content_words or any(w in clean_words for w in content_words):
                return True

        if m_content in clean:
            return True

        return False

    def discard_memory(self, target: Any) -> bool:
        with self._lock:
            if not target:
                return False
            memories = self.data.get("memories", [])
            initial_len = len(memories)
            self.data["memories"] = [
                m for m in memories
                if not self._matches_memory_target(m, target)
            ]
            if len(self.data["memories"]) < initial_len:
                self.save()
                return True
            return False

    # Dynamics
    def add_dynamic(self, members: List[str], relation: str) -> None:
        with self._lock:
            dynamics = self.data.setdefault("dynamics", [])
            entry = {
                "members": [m.strip() for m in members if m.strip()],
                "relation": relation.strip(),
                "created_at": _utc_iso_now(),
            }
            dynamics.append(entry)
            self.save()

    def _matches_dynamic_target(self, d: Dict[str, Any], target: Any) -> bool:
        d_members = [str(m).strip().lower() for m in d.get("members", [])]
        d_relation = str(d.get("relation", "")).strip().lower()

        if isinstance(target, dict):
            t_members = [str(m).strip().lower() for m in target.get("members", []) if str(m).strip()]
            t_relation = str(target.get("relation", "")).strip().lower()
            if t_members and any(m in d_members for m in t_members):
                if not t_relation or t_relation in d_relation or d_relation in t_relation:
                    return True
            if t_relation and (t_relation in d_relation or d_relation in t_relation):
                return True
            return False

        clean = str(target).strip().lower()
        if not clean:
            return False

        prefix_match = re.match(r"^(?:-\s*)?(?:\(([^)]+)\)|([^:]+)):\s*(.*)$", clean)
        if prefix_match:
            cand_members_str = (prefix_match.group(1) or prefix_match.group(2) or "").strip().lower()
            cand_relation = prefix_match.group(3).strip().lower()
            cand_members = [m.strip() for m in re.split(r"&|and|,", cand_members_str) if m.strip()]
            if any(m in d_members for m in cand_members):
                if not cand_relation or cand_relation in d_relation or d_relation in cand_relation:
                    return True

        if clean == d_relation or clean in d_relation or d_relation in clean:
            return True

        if any(clean == m or clean in m or m in clean for m in d_members):
            return True

        return False

    def discard_dynamic(self, target: Any) -> bool:
        with self._lock:
            if not target:
                return False
            dynamics = self.data.get("dynamics", [])
            initial_len = len(dynamics)
            self.data["dynamics"] = [
                d for d in dynamics
                if not self._matches_dynamic_target(d, target)
            ]
            if len(self.data["dynamics"]) < initial_len:
                self.save()
                return True
            return False

    # Inside Jokes
    def add_inside_joke(self, title: str, context: str) -> None:
        with self._lock:
            jokes = self.data.setdefault("inside_jokes", [])
            entry = {
                "title": title.strip(),
                "context": context.strip(),
                "created_at": _utc_iso_now(),
            }
            jokes.append(entry)
            self.save()

    def _matches_inside_joke_target(self, j: Dict[str, Any], target: Any) -> bool:
        j_title = str(j.get("title", "")).strip().lower()
        j_context = str(j.get("context", "")).strip().lower()

        if isinstance(target, dict):
            t_title = str(target.get("title", "")).strip().lower()
            t_context = str(target.get("context", "")).strip().lower()
            if t_title and (t_title in j_title or j_title in t_title):
                return True
            if t_context and (t_context in j_context or j_context in t_context):
                return True
            return False

        clean = str(target).strip().lower()
        if not clean:
            return False

        prefix_match = re.match(r"^(?:-\s*)?(?:\(([^)]+)\)|([^:]+)):\s*(.*)$", clean)
        if prefix_match:
            cand_title = (prefix_match.group(1) or prefix_match.group(2) or "").strip().lower()
            cand_ctx = prefix_match.group(3).strip().lower()
            if cand_title and (cand_title in j_title or j_title in cand_title):
                return True
            if cand_ctx and (cand_ctx in j_context or j_context in cand_ctx):
                return True

        if clean == j_title or clean in j_title or j_title in clean:
            return True
        if clean in j_context or j_context in clean:
            return True

        return False

    def discard_inside_joke(self, target: Any) -> bool:
        with self._lock:
            if not target:
                return False
            jokes = self.data.get("inside_jokes", [])
            initial_len = len(jokes)
            self.data["inside_jokes"] = [
                j for j in jokes
                if not self._matches_inside_joke_target(j, target)
            ]
            if len(self.data["inside_jokes"]) < initial_len:
                self.save()
                return True
            return False

    # Generic Discard
    def discard_any(self, target: Any) -> bool:
        with self._lock:
            r1 = self.discard_memory(target)
            r2 = self.discard_dynamic(target)
            r3 = self.discard_inside_joke(target)
            return r1 or r2 or r3

    def apply_forget(self, forget_spec: Dict[str, Any]) -> Dict[str, int]:
        """Applies a forget specification, thoroughly removing or updating matching memories."""
        with self._lock:
            discarded_facts = 0
            updated_facts = 0
            discarded_dyn = 0
            discarded_jokes = 0

            if forget_spec.get("clear_all"):
                initial_facts = len(self.data.get("memories", []))
                initial_dyn = len(self.data.get("dynamics", []))
                initial_jokes = len(self.data.get("inside_jokes", []))
                self.clear()
                return {
                    "discarded_facts": initial_facts,
                    "updated_facts": 0,
                    "discarded_dynamics": initial_dyn,
                    "discarded_jokes": initial_jokes,
                }

            # 1. Facts to Discard
            for item in forget_spec.get("facts_to_discard", []):
                if self.discard_memory(item):
                    discarded_facts += 1

            # 2. Facts to Update
            for item in forget_spec.get("facts_to_update", []):
                if isinstance(item, dict):
                    topic = str(item.get("topic", "")).strip()
                    content = str(item.get("content", "")).strip()
                    if topic:
                        if not content:
                            if self.discard_memory(topic):
                                discarded_facts += 1
                        else:
                            if self.update_memory(topic, content):
                                updated_facts += 1

            # 3. Dynamics to Discard
            for item in forget_spec.get("dynamics_to_discard", []):
                if self.discard_dynamic(item):
                    discarded_dyn += 1

            # 4. Jokes to Discard
            for item in forget_spec.get("jokes_to_discard", []):
                if self.discard_inside_joke(item):
                    discarded_jokes += 1

            # 5. Raw Targets fallback
            for item in forget_spec.get("raw_targets", []):
                if self.discard_any(item):
                    discarded_facts += 1

            return {
                "discarded_facts": discarded_facts,
                "updated_facts": updated_facts,
                "discarded_dynamics": discarded_dyn,
                "discarded_jokes": discarded_jokes,
            }
    def clear(self) -> None:
        with self._lock:
            self.data["memories"] = []
            self.data["dynamics"] = []
            self.data["inside_jokes"] = []
            self.save()

    def get_all_memories(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {
                "memories": list(self.data.get("memories", [])),
                "dynamics": list(self.data.get("dynamics", [])),
                "inside_jokes": list(self.data.get("inside_jokes", [])),
            }

    def set_last_curated_at(self, timestamp_iso: str) -> None:
        with self._lock:
            self.data["last_curated_at"] = timestamp_iso
            self.save()

    def get_last_curated_at(self) -> str:
        with self._lock:
            return self.data.get("last_curated_at", "")

    def format_for_context(self) -> str:
        with self._lock:
            memories = list(self.data.get("memories", []))
            dynamics = list(self.data.get("dynamics", []))
            jokes = list(self.data.get("inside_jokes", []))

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
