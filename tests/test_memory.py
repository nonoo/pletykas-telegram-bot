import glob
import json
import os
import tempfile
from unittest.mock import patch
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
        assert all_m[0]["updated_at"] == all_m[0]["created_at"]
        assert all_m[1]["updated_at"] == all_m[1]["created_at"]

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

        with patch("memory._utc_iso_now", return_value="2026-01-01T00:00:00Z"):
            mm.add_memory("Alice", "Lives in Berlin")
        with patch("memory._utc_iso_now", return_value="2026-01-02T00:00:00Z"):
            assert mm.update_memory("Alice", "Moved to Munich") is True
        assert mm.update_memory("Nonexistent", "Does not matter") is False

        all_m = mm.get_all_memories()["memories"]
        assert len(all_m) == 1
        assert all_m[0]["topic"] == "Alice"
        assert all_m[0]["content"] == "Moved to Munich"
        # update resets updated_at but keeps the original created_at
        assert all_m[0]["created_at"] == "2026-01-01T00:00:00Z"
        assert all_m[0]["updated_at"] == "2026-01-02T00:00:00Z"


def test_memory_load_backfills_updated_at_from_created_at():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        # Legacy file: one fact with created_at but no updated_at, one with neither
        with open(mf, "w", encoding="utf-8") as f:
            json.dump({
                "version": 1,
                "memories": [
                    {"id": "mem_legacy", "topic": "Alice", "content": "Lives in Berlin",
                     "created_at": "2026-01-01T00:00:00Z"},
                    {"topic": "Bob", "content": "Cyclist"},
                ],
                "dynamics": [],
                "inside_jokes": [],
            }, f)
        mm = MemoryManager(mf)
        mm.load()
        all_m = mm.get_all_memories()["memories"]
        assert "id" not in all_m[0]
        assert all_m[0]["updated_at"] == "2026-01-01T00:00:00Z"
        assert all_m[1]["updated_at"] == ""


def test_memory_pop_and_append_entry():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        with patch("memory._utc_iso_now", return_value="2026-01-01T00:00:00Z"):
            mm.add_memory("Alice", "Lives in Berlin")
        mm.add_dynamic(["Alice", "Bob"], "Friends")
        mm.add_inside_joke("Joke1", "Ctx")

        popped = mm.pop_memory("Alice")
        assert popped is not None
        assert popped["topic"] == "Alice"
        assert popped["created_at"] == "2026-01-01T00:00:00Z"
        assert popped["updated_at"] == "2026-01-01T00:00:00Z"
        assert mm.get_all_memories()["memories"] == []
        assert mm.pop_memory("Alice") is None

        # Pop persists to disk
        mm2 = MemoryManager(mf)
        mm2.load()
        assert mm2.get_all_memories()["memories"] == []

        popped_dyn = mm.pop_dynamic("Alice")
        assert popped_dyn is not None
        assert popped_dyn["relation"] == "Friends"
        assert mm.get_all_memories()["dynamics"] == []
        assert mm.pop_dynamic("nobody") is None

        popped_joke = mm.pop_inside_joke("Joke1")
        assert popped_joke is not None
        assert popped_joke["context"] == "Ctx"
        assert mm.pop_inside_joke("Joke1") is None

        # append_entry copies the entry VERBATIM (no timestamp rewrite)
        target = MemoryManager(os.path.join(td, "deep.json"))
        target.load()
        target.append_entry("memories", popped)
        target.append_entry("dynamics", {"members": ["A"], "relation": "x"})
        got = target.get_all_memories()
        assert got["memories"] == [popped]
        assert got["memories"][0]["created_at"] == "2026-01-01T00:00:00Z"
        assert got["dynamics"] == [{"members": ["A"], "relation": "x"}]

        # Mutating the source dict after append does not affect the stored copy
        popped["content"] = "Changed"
        assert target.get_all_memories()["memories"][0]["content"] == "Lives in Berlin"

        with pytest.raises(ValueError):
            target.append_entry("unknown_section", {})


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


def test_forget_targets_do_not_over_delete():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)

        mm.add_memory('Norbert "Nonoo" Varga', "On 2026-09-30 at 09:05 said he had already woken up and worked but was on the retyó")
        mm.add_memory('Norbert "Nonoo" Varga', "On 2026-10-08 at 10:02 tested Pletykas with 'pletyi teszt'")
        mm.add_memory("Pletykas", "On 2026-10-08 at 10:03 responded to Norbert, the test was successful")
        mm.add_dynamic(["Norbert", "Pletykas"], "Norbert teases Pletykas about the retyó")

        # A long structured target removes ONLY the entry it describes: sharing the
        # topic plus one common word ("2026", "was", ...) must not match anything else.
        target = '(Norbert "Nonoo" Varga): On 2026-10-08 at 10:02 tested Pletykas with \'pletyi teszt\''
        assert mm.discard_memory(target) is True
        facts = mm.get_all_memories()["memories"]
        assert len(facts) == 2
        assert all("pletyi teszt" not in f["content"] for f in facts)

        # Same class of structured target must not nuke a dynamic by member name alone
        assert mm.discard_dynamic("(Norbert & Pletykas): something completely unrelated") is False
        assert len(mm.get_all_memories()["dynamics"]) == 1

        # Quoted topics still match exactly
        mm.add_memory("Deepmem teszt téma", "Ez egy régi, archívumba való teszt bejegyzés")
        assert mm.discard_memory('"Deepmem teszt téma"') is True
        assert all(f["topic"] != "Deepmem teszt téma" for f in mm.get_all_memories()["memories"])

        # The unstructured topic+phrase heuristic still works when EVERY content word is present
        mm.add_memory("Alice", "Lives in Berlin")
        assert mm.discard_memory("Alice lives in Berlin") is True
        assert all(f["topic"] != "Alice" for f in mm.get_all_memories()["memories"])


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
    assert cleaned["memories"][0]["updated_at"] == cleaned["memories"][0]["created_at"]
    assert "id" not in cleaned["dynamics"][0]
    assert "id" not in cleaned["inside_jokes"][0]

    # Explicit updated_at is preserved
    is_valid, err, cleaned2 = validate_memory_dict({
        "memories": [
            {"topic": "Bob", "content": "Cyclist",
             "created_at": "2026-01-01T00:00:00Z",
             "updated_at": "2026-02-02T00:00:00Z"}
        ]
    })
    assert is_valid is True
    assert cleaned2["memories"][0]["created_at"] == "2026-01-01T00:00:00Z"
    assert cleaned2["memories"][0]["updated_at"] == "2026-02-02T00:00:00Z"

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
def test_memory_thread_safety():
    import threading
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        errors = []

        def worker(thread_idx: int):
            try:
                for i in range(20):
                    mm.add_memory(f"Topic_{thread_idx}_{i}", f"Fact content {i}")
                    mm.add_dynamic([f"User_{thread_idx}", f"User_{i}"], "Colleagues")
                    mm.add_inside_joke(f"Joke_{thread_idx}_{i}", "Funny lore")
                    _ = mm.get_all_memories()
                    _ = mm.format_for_context()
                    mm.create_backup()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors encountered: {errors}"
        all_m = mm.get_all_memories()
        assert len(all_m["memories"]) == 6 * 20
        assert len(all_m["dynamics"]) == 6 * 20
        assert len(all_m["inside_jokes"]) == 6 * 20
def test_memory_file_parse_error_quits():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        with open(mf, "w", encoding="utf-8") as f:
            f.write("{invalid json memory file [[")
        mm = MemoryManager(mf)
        with pytest.raises(SystemExit) as exc_info:
            mm.load()
        assert exc_info.value.code == 1

def test_memory_apply_forget_and_target_matching():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        mm.add_memory("Alice", "Lives in Berlin and writes Rust")
        mm.add_memory("Bob", "Loves pineapple pizza and works in Munich")
        mm.add_dynamic(["Alice", "Bob"], "Former roommates in Berlin")
        mm.add_inside_joke("Berlin Wall of Code", "Debugging marathon in Berlin")
        mm.add_inside_joke("Pineapple Gate", "Pizza topping debate")

        # Test partial update on Alice (forgetting Berlin, keeping Rust)
        # Test discarding pineapple joke
        forget_spec = {
            "facts_to_update": [{"topic": "Alice", "content": "Writes Rust"}],
            "jokes_to_discard": ["Pineapple Gate"],
        }
        summary = mm.apply_forget(forget_spec)
        assert summary["updated_facts"] == 1
        assert summary["discarded_jokes"] == 1

        memories = mm.get_all_memories()["memories"]
        assert len(memories) == 2
        alice_mem = next(m for m in memories if m["topic"] == "Alice")
        assert alice_mem["content"] == "Writes Rust"

        # Test formatted discard: "- (Bob): pineapple"
        summary2 = mm.apply_forget({"facts_to_discard": ["- (Bob): pineapple"]})
        assert summary2["discarded_facts"] == 1
        assert len(mm.get_all_memories()["memories"]) == 1

        # Test dynamic discard with members and relation
        summary3 = mm.apply_forget({"dynamics_to_discard": ["Alice & Bob: Berlin"]})
        assert summary3["discarded_dynamics"] == 1
        assert len(mm.get_all_memories()["dynamics"]) == 0

        # Test dict target in discard_memory and discard_inside_joke
        assert mm.discard_inside_joke({"title": "Berlin Wall of Code"}) is True
        assert len(mm.get_all_memories()["inside_jokes"]) == 0


def test_memory_apply_forget_clear_all():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        mm.add_memory("Alice", "Lives in Berlin")
        mm.add_dynamic(["Alice", "Bob"], "Friends")
        mm.add_inside_joke("Joke", "Context")

        summary = mm.apply_forget({"clear_all": True})
        assert summary["discarded_facts"] == 1
        assert summary["discarded_dynamics"] == 1
        assert summary["discarded_jokes"] == 1

        all_m = mm.get_all_memories()
        assert len(all_m["memories"]) == 0
        assert len(all_m["dynamics"]) == 0
        assert len(all_m["inside_jokes"]) == 0


def test_memory_discard_any_does_not_short_circuit():
    with tempfile.TemporaryDirectory() as td:
        mf = os.path.join(td, "mem.json")
        mm = MemoryManager(mf)
        mm.load()

        # All three reference "Alice"
        mm.add_memory("Alice", "Engineer")
        mm.add_dynamic(["Alice", "Bob"], "Coworkers")
        mm.add_inside_joke("Alice's Typo", "Hilarious bug")

        # discard_any("Alice") should remove from all three without stopping at facts
        res = mm.discard_any("Alice")
        assert res is True

        all_m = mm.get_all_memories()
        assert len(all_m["memories"]) == 0
        assert len(all_m["dynamics"]) == 0
        assert len(all_m["inside_jokes"]) == 0
