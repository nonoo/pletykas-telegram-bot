"""Deep-memory embedding sidecar: vector cache + cosine retrieval.

Pure logic + file IO (stdlib only — no params/llm/handlers imports). Callers
provide the embed function, keeping this module dependency-free and testable.
"""
import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Cosine threshold, calibrated 2026-10-08 against the live store with gemini-embedding-2:
# unrelated queries scored <=0.596, relevant ones >=0.688 (n=10). Re-tune if the embed model changes.
DEEPMEM_MIN_SCORE = 0.62
DEEPMEM_TOP_K = 3


def entry_text(entry: Dict[str, Any], section: str) -> str:
    """Canonical embeddable text for an entry (label + body, as shown to the model)."""
    if section == "memories":
        text = f"{entry.get('topic', '')}: {entry.get('content', '')}"
    elif section == "dynamics":
        members = " & ".join(str(m) for m in entry.get("members", []))
        text = f"{members}: {entry.get('relation', '')}"
    elif section == "inside_jokes":
        text = f"{entry.get('title', '')}: {entry.get('context', '')}"
    else:
        text = ""
    return text.strip()


def entry_hash(entry: Dict[str, Any], section: str) -> str:
    """sha256 hex over section + text + created_at (created_at disambiguates duplicate texts)."""
    raw = f"{section}\0{entry_text(entry, section)}\0{entry.get('created_at', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DeepMemoryIndex:
    """Sidecar vector cache for the cold-tier memory store.

    Vectors are keyed by content hash (entries are ID-free by invariant), so
    edits/removals naturally invalidate stale vectors during `sync`.
    """

    def __init__(
        self,
        sidecar_path: str = "pletykas-deepmemory-embeddings.json",
        embed_model: str = "google/gemini-embedding-2",
    ):
        # The literal default must match the params.py default (model_embed_name).
        # Production passes the configured model explicitly, so a future model
        # swap invalidates stale vectors via the model-mismatch rule in load().
        self._embed_model = embed_model
        self._sidecar_path = sidecar_path
        self._lock = threading.RLock()
        self._vectors: Dict[str, List[float]] = {}
        self._model = ""

    def load(self) -> None:
        with self._lock:
            if not os.path.exists(self._sidecar_path):
                return
            try:
                with open(self._sidecar_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("sidecar root is not an object")
                vectors = data.get("vectors", {})
                if not isinstance(vectors, dict):
                    raise ValueError("sidecar 'vectors' is not an object")
                model = data.get("model", "")
                if model != self._embed_model:
                    logger.warning(
                        "Deep-memory embeddings sidecar model mismatch (%r != %r); discarding cache",
                        model,
                        self._embed_model,
                    )
                    self._vectors = {}
                else:
                    self._vectors = {
                        str(key): [float(v) for v in vec]
                        for key, vec in vectors.items()
                        if isinstance(vec, list)
                    }
                self._model = self._embed_model
            except Exception as e:
                logger.warning(
                    "Discarding malformed deep-memory embeddings sidecar (%s): %s",
                    self._sidecar_path,
                    e,
                )
                self._vectors = {}
                self._model = self._embed_model

    def save(self) -> None:
        # No backup rotation: this cache is fully regenerable from the deep store.
        with self._lock:
            data = {"model": self._model or self._embed_model, "vectors": self._vectors}
            directory = os.path.dirname(self._sidecar_path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-deepmem-emb-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp_path, self._sidecar_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def sync(
        self,
        entries: List[Tuple[str, Dict[str, Any]]],
        embed_fn: Callable[[List[str]], List[List[float]]],
    ) -> int:
        """Drops vectors for vanished entries and embeds missing ones in ONE call.

        `embed_fn` is SYNC (List[str] -> List[List[float]]); the caller bridges
        async embedding. Returns the number of newly embedded entries, or 0 on
        failure (old state kept).
        """
        with self._lock:
            current = {
                entry_hash(entry, section): entry_text(entry, section)
                for section, entry in entries
            }
            pruned = {h: v for h, v in self._vectors.items() if h in current}
            dropped = len(self._vectors) - len(pruned)
            self._vectors = pruned
            missing = [(h, text) for h, text in current.items() if text and h not in self._vectors]
            if not missing:
                self._model = self._embed_model
                if dropped:
                    self.save()
                return 0
            texts = [text for _, text in missing]
            try:
                vectors = embed_fn(texts)
            except Exception as e:
                logger.warning("Deep-memory embedding sync failed: %s", e)
                return 0
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                logger.warning(
                    "Deep-memory embedding sync got %s vectors for %d texts; keeping old state",
                    len(vectors) if isinstance(vectors, list) else "invalid",
                    len(texts),
                )
                return 0
            stored = 0
            for (h, _), vec in zip(missing, vectors):
                if isinstance(vec, list) and vec:
                    self._vectors[h] = [float(v) for v in vec]
                    stored += 1
            self._model = self._embed_model
            self.save()
            return stored

    def count_vectors(self) -> int:
        """Number of cached vectors (status reporting)."""
        with self._lock:
            return len(self._vectors)

    def query(
        self,
        query_vector: List[float],
        entries: List[Tuple[str, Dict[str, Any]]],
        top_k: int = DEEPMEM_TOP_K,
        min_score: float = DEEPMEM_MIN_SCORE,
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Cosine-similarity top-k over stored vectors. Pure function of inputs (no IO)."""
        with self._lock:
            q = [float(v) for v in (query_vector or [])]
            q_norm = math.sqrt(sum(v * v for v in q))
            if q_norm <= 0.0:
                return []
            scored: List[Tuple[float, str, Dict[str, Any]]] = []
            for section, entry in entries:
                vec = self._vectors.get(entry_hash(entry, section))
                if not vec or len(vec) != len(q):
                    continue
                n = math.sqrt(sum(v * v for v in vec))
                if n <= 0.0:
                    continue
                score = sum(a * b for a, b in zip(q, vec)) / (q_norm * n)
                if score >= min_score:
                    scored.append((score, section, entry))
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[:top_k]

    def format_hits(self, hits: List[Tuple[float, str, Dict[str, Any]]]) -> str:
        """Renders hits in MemoryManager.format_for_context line style; '' when empty."""
        if not hits:
            return ""
        lines = ["[Recalled Deep Memory]"]
        for _, section, entry in hits:
            if section == "memories":
                label, text = entry.get("topic", ""), entry.get("content", "")
            elif section == "dynamics":
                label, text = " & ".join(str(m) for m in entry.get("members", [])), entry.get("relation", "")
            else:
                label, text = entry.get("title", ""), entry.get("context", "")
            if label or text:
                lines.append(f"- ({label}): {text}")
        return "\n".join(lines)
