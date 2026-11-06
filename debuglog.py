"""In-memory debug log: always captured, byte-capped, downloadable by the admin."""

import threading
from typing import List

# "under 1 megabyte" -> 1,000,000 bytes so the bound holds for either MB reading.
DEBUG_LOG_MAX_BYTES = 1_000_000


class DebugLog:
    """Thread-safe in-memory log capped by dropping oldest whole lines.

    append() never caps: cap() runs lazily at flow ends (after replying,
    curation, embedding refresh) and inside dump(), never per line.
    """

    def __init__(self, max_bytes: int = DEBUG_LOG_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._lines: List[str] = []   # stored without trailing newlines, oldest first
        self._size_bytes = 0          # sum of len(line.encode("utf-8")) + 1 per line
        self._lock = threading.Lock()

    def append(self, entry: str) -> None:
        """Appends a (possibly multi-line) entry verbatim. No size check here."""
        with self._lock:
            for line in entry.split("\n"):
                self._lines.append(line)
                self._size_bytes += len(line.encode("utf-8")) + 1

    def _cap_locked(self) -> None:
        """Drops oldest whole lines until the log is strictly below the limit. Caller holds the lock."""
        while self._lines and self._size_bytes >= self._max_bytes:
            self._size_bytes -= len(self._lines.pop(0).encode("utf-8")) + 1

    def cap(self) -> None:
        """Drops oldest whole lines until the log is strictly below the limit."""
        with self._lock:
            self._cap_locked()

    def dump(self) -> str:
        """Returns the log text, capping first so the result is always under the limit."""
        with self._lock:
            self._cap_locked()
            return "\n".join(self._lines)
