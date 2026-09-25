import glob
import os
import tempfile
import pytest
from memory import MemoryManager, validate_memory_dict


def test_memory_defaults_and_formatting():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()
        assert mm.format_for_context() == "<memory>None</memory>"


def test_memory_add_and_retention_no_ids():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        mm.add_memory("Alice", "Lives in Berlin")
        mm.add_memory("Bob", "Enjoys coffee")

        all_m = mm.get_all_memories()["memories"]
        assert len(all_m) == 2
        assert "id" not in all_m[0]
        assert all_m[0]["topic"] == "Alice"
        assert all_m[0]["content"] == "Lives in Berlin"
        assert "id" not in all_m[1]
        assert "updated_at" not in all_m[0]
        assert "updated_at" not in all_m[1]

        # Dynamics without IDs
        mm.add_dynamic(["Alice", "Bob"], "Old friends")
        all_d = mm.get_all_memories()["dynamics"]
        assert len(all_d) == 1
        assert "id" not in all_d[0]
        assert all_d[0]["members"] == ["Alice", "Bob"]
        assert all_d[0]["relation"] == "Old friends"

        # Jokes without IDs
        mm.add_inside_joke("Tabs", "Indentation war")
        all_j = mm.get_all_memories()["inside_jokes"]
        assert len(all_j) == 1
        assert "id" not in all_j[0]
        assert all_j[0]["title"] == "Tabs"


def test_memory_updates_by_topic():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        mm.add_memory("Alice", "Lives in Berlin")
        assert mm.update_memory("Alice", "Moved to Munich") is True
        assert mm.update_memory("Nonexistent", "Does not matter") is False

        all_m = mm.get_all_memories()["memories"]
        assert len(all_m) == 1
        assert all_m[0]["topic"] == "Alice"
        assert all_m[0]["content"] == "Moved to Munich"
        assert "updated_at" not in all_m[0]


def test_memory_discard_by_content_or_topic():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        mm.add_memory("Alice", "Lives in Berlin")
        mm.add_dynamic(["Alice", "Bob"], "Code reviewers")
        mm.add_inside_joke("Tabs vs Spaces", "Indentation dispute")

        # Discard fact by topic
        assert mm.discard_memory("Alice") is True
        assert mm.discard_memory("Alice") is False
        assert len(mm.get_all_memories()["memories"]) == 0

        # Discard dynamic by relation keyword
        assert mm.discard_dynamic("reviewers") is True
        assert len(mm.get_all_memories()["dynamics"]) == 0

        # Discard joke by title
        assert mm.discard_inside_joke("Tabs") is True
        assert len(mm.get_all_memories()["inside_jokes"]) == 0


def test_memory_discard_any():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        mm.add_memory("Topic", "Fact content")
        mm.add_dynamic(["UserA", "UserB"], "Colleagues")
        mm.add_inside_joke("JokeTitle", "Joke context")

        assert mm.discard_any("Fact content") is True
        assert mm.discard_any("Colleagues") is True
        assert mm.discard_any("JokeTitle") is True
        assert mm.discard_any("fake_99") is False

        all_m = mm.get_all_memories()
        assert len(all_m["memories"]) == 0
        assert len(all_m["dynamics"]) == 0
        assert len(all_m["inside_jokes"]) == 0


def test_memory_format_for_context():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        mm.add_memory("Alice", "Rust programmer")
        mm.add_dynamic(["Alice", "Bob"], "Code reviewers")
        mm.add_inside_joke("Async Soup", "Debugging deadlock")

        ctx = mm.format_for_context()
        assert "<memory>" in ctx
        assert "</memory>" in ctx
        assert "[Facts]" in ctx
        assert "- (Alice): Rust programmer" in ctx
        assert "[Group Dynamics]" in ctx
        assert "- (Alice & Bob): Code reviewers" in ctx
        assert "[Inside Jokes & Lore]" in ctx
        assert "- (Async Soup): Debugging deadlock" in ctx
        # Ensure no ID tags like [mem_1], [dyn_1], [joke_1]
        assert "[mem_" not in ctx
        assert "[dyn_" not in ctx
        assert "[joke_" not in ctx


def test_validate_memory_dict():
    # Valid data with legacy IDs (should be stripped)
    valid_data = {
        "version": 1,
        "memories": [
            {"id": "legacy_1", "topic": "Alice", "content": "Engineer"}
        ],
        "dynamics": [
            {"id": "legacy_2", "members": ["Alice", "Bob"], "relation": "Coworkers"}
        ],
        "inside_jokes": [
            {"id": "legacy_3", "title": "Coffee", "context": "Never enough coffee"}
        ]
    }
    is_valid, err, cleaned = validate_memory_dict(valid_data)
    assert is_valid is True
    assert err == ""
    assert cleaned is not None
    assert "id" not in cleaned["memories"][0]
    assert cleaned["memories"][0]["topic"] == "Alice"
    assert "updated_at" not in cleaned["memories"][0]
    assert "id" not in cleaned["dynamics"][0]
    assert "id" not in cleaned["inside_jokes"][0]

    # Non-dict
    is_valid, err, _ = validate_memory_dict(["not", "a", "dict"])
    assert is_valid is False
    assert "dict" in err.lower()

    # Invalid list type
    is_valid, err, _ = validate_memory_dict({"memories": "not a list"})
    assert is_valid is False
    assert "list" in err.lower()

    # Missing required field in memories
    is_valid, err, _ = validate_memory_dict({"memories": [{"topic": "Alice"}]})
    assert is_valid is False
    assert "content" in err.lower()


def test_memory_backup_and_pruning():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()
        mm.add_memory("Topic", "Content")

        for _ in range(25):
            mm.create_backup()

        pattern = os.path.join(td, "mem-*.json.bak")
        backups = glob.glob(pattern)
        assert len(backups) <= 20


def test_memory_clear():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        mm.add_memory("Topic", "Content")
        mm.add_dynamic(["A", "B"], "Friends")
        mm.add_inside_joke("Joke", "Text")

        mm.clear()
        assert mm.format_for_context() == "<memory>None</memory>"
        all_m = mm.get_all_memories()
        assert all_m["memories"] == []
        assert all_m["dynamics"] == []
        assert all_m["inside_jokes"] == []
