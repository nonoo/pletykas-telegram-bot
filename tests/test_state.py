import os
import tempfile
import time
from datetime import datetime, time as dtime
import zoneinfo
import pytest
from state import CHAT_HISTORY_SIZE, MEMORY_HISTORY_SIZE, StateManager


def test_state_load_defaults():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf, group_chat_id=-100123)
        sm.load()
        assert sm.get_group_chat_id() == -100123
        assert sm.get_language() == "English"
        assert sm.get_timezone() == "UTC"
        assert sm.get_talkativeness() == 5
        assert sm.get_cooldown_sec() == 5
        assert sm.is_search_grounding_active() is True
        assert sm.is_debug_mode() is False
        assert sm.get_nicknames() == ["pletyi", "pletyo"]
        assert os.path.exists(sf)


def test_state_atomic_save_reload():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf, group_chat_id=-100)
        sm.load()
        sm.set_language("English")
        sm.set_talkativeness(9)
        sm.set_cooldown_sec(8)
        sm.set_search_grounding_active(True)
        sm.set_debug_mode(True)
        sm.set_nicknames(["pletyka", "@pleti", "pletyka"])

        sm2 = StateManager(sf)
        sm2.load()
        assert sm2.get_language() == "English"
        assert sm2.get_talkativeness() == 9
        assert sm2.get_cooldown_sec() == 8
        assert sm2.is_search_grounding_active() is True
        assert sm2.is_debug_mode() is True
        assert sm2.get_nicknames() == ["pletyka", "pleti"]


def test_state_nicknames_in_prompt():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()
        assert "Your nicknames: pletyi, pletyo." in sm.get_effective_system_prompt()
        sm.set_nicknames([])
        assert "Your nicknames:" not in sm.get_effective_system_prompt()
        sm.set_nicknames(["pleti", "kis pletyka"])
        assert "Your nicknames: pleti, kis pletyka." in sm.get_effective_system_prompt()

def test_state_timezone():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()
        assert sm.set_timezone("America/New_York") is True
        assert sm.get_timezone() == "America/New_York"
        assert sm.set_timezone("Invalid/Fake_Zone") is False
        assert sm.get_timezone() == "America/New_York"

def test_state_system_prompt():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()
        eff = sm.get_effective_system_prompt()
        assert "Always communicate in English." in eff
        assert "Use Telegram HTML markdown. Only use the following HTML tags b, i, u, s, a, code, blockquote. Do not use any other HTML tags. Do not use LaTeX for formatting." in eff
        sm.set_language("Hungarian")
        assert "Always communicate in Hungarian." in sm.get_effective_system_prompt()


def test_state_sleep_schedule_wraparound(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()

        # Overnight schedule: 23:00 to 07:00
        sm.set_sleep_schedule(enabled=True, sleep_start="23:00", sleep_end="07:00")

        # Mock current time at 23:30 (should be sleeping)
        class MockDT2330:
            def time(self):
                return dtime(23, 30)

        monkeypatch.setattr(sm, "get_current_time", lambda: MockDT2330())
        assert sm.is_sleeping() is True

        # Mock current time at 03:15 (should be sleeping)
        class MockDT0315:
            def time(self):
                return dtime(3, 15)

        monkeypatch.setattr(sm, "get_current_time", lambda: MockDT0315())
        assert sm.is_sleeping() is True

        # Mock current time at 12:00 (awake)
        class MockDT1200:
            def time(self):
                return dtime(12, 0)

        monkeypatch.setattr(sm, "get_current_time", lambda: MockDT1200())
        assert sm.is_sleeping() is False


def test_state_sleep_schedule_daytime(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()

        # Daytime nap: 13:00 to 15:00
        sm.set_sleep_schedule(enabled=True, sleep_start="13:00", sleep_end="15:00")

        class MockDT1400:
            def time(self):
                return dtime(14, 0)

        monkeypatch.setattr(sm, "get_current_time", lambda: MockDT1400())
        assert sm.is_sleeping() is True

        class MockDT1600:
            def time(self):
                return dtime(16, 0)

        monkeypatch.setattr(sm, "get_current_time", lambda: MockDT1600())
        assert sm.is_sleeping() is False

def test_state_spontaneous_settings():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()

        assert sm.is_spontaneous_enabled() is True
        sm.set_spontaneous_settings(False)
        assert sm.is_spontaneous_enabled() is False
        sm.set_spontaneous_settings(True)
        assert sm.is_spontaneous_enabled() is True
def test_state_talkativeness_and_cooldown_clamping():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()

        sm.set_talkativeness(0)
        assert sm.get_talkativeness() == 1
        sm.set_talkativeness(15)
        assert sm.get_talkativeness() == 10

        sm.set_cooldown_sec(-5)
        assert sm.get_cooldown_sec() == 0
        sm.set_cooldown_sec(10)
        assert sm.get_cooldown_sec() == 10

def test_state_history_sliding_window():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()

        # Append 25 chat messages (CHAT_HISTORY_SIZE = 20)
        for i in range(25):
            sm.append_chat_message({"id": i, "text": f"Msg {i}"})

        history = sm.get_chat_history()
        assert len(history) == CHAT_HISTORY_SIZE
        assert history[0]["id"] == 5
        assert history[-1]["id"] == 24

        # Append 35 memory messages (MEMORY_HISTORY_SIZE = 30)
        for i in range(35):
            sm.append_memory_message({"id": i, "text": f"Mem {i}"})

        mem_history = sm.get_memory_history()
        assert len(mem_history) == MEMORY_HISTORY_SIZE
        assert mem_history[0]["id"] == 5
        assert mem_history[-1]["id"] == 34


def test_state_supergroup_migration():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf, group_chat_id=-100)
        sm.load()
        assert sm.get_group_chat_id() == -100
        sm.set_group_chat_id(-100999888)
        assert sm.get_group_chat_id() == -100999888

        # Verify persisted
        sm2 = StateManager(sf)
        sm2.load()
        assert sm2.get_group_chat_id() == -100999888
def test_state_image_interpretation_large_model():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()
        # Enabled by default
        assert sm.is_image_interpretation_large_model() is True

        # Toggle off
        sm.set_image_interpretation_large_model(False)
        assert sm.is_image_interpretation_large_model() is False

        # Reload from disk
        sm2 = StateManager(sf)
        sm2.load()
        assert sm2.is_image_interpretation_large_model() is False

        # Toggle back on
        sm2.set_image_interpretation_large_model(True)
        assert sm2.is_image_interpretation_large_model() is True


def test_state_search_small_model():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()
        # Disabled by default
        assert sm.is_search_small_model() is False

        # Toggle on
        sm.set_search_small_model(True)
        assert sm.is_search_small_model() is True

        # Reload from disk
        sm2 = StateManager(sf)
        sm2.load()
        assert sm2.is_search_small_model() is True

        # Toggle back off
        sm2.set_search_small_model(False)
        assert sm2.is_search_small_model() is False

def test_state_spontaneous_settings():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()

        spont = sm.get_spontaneous_settings()
        assert spont["enabled"] is True
        assert spont["min_hours"] == 2.0
        assert spont["max_hours"] == 4.0

        sm.set_spontaneous_interval(3.5, 6.0)
        assert sm.get_spontaneous_settings()["min_hours"] == 3.5
        assert sm.get_spontaneous_settings()["max_hours"] == 6.0

        sm.set_spontaneous_settings(enabled=False)
        assert sm.is_spontaneous_enabled() is False

        # Reload from disk
        sm2 = StateManager(sf)
        sm2.load()
        assert sm2.is_spontaneous_enabled() is False
        assert sm2.get_spontaneous_settings()["min_hours"] == 3.5
        assert sm2.get_spontaneous_settings()["max_hours"] == 6.0

def test_state_spontaneous_next_fire_time():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()

        # Default is None
        assert sm.get_spontaneous_next_fire_time() is None

        # Persist a datetime
        dt = datetime(2026, 9, 25, 20, 15, 30, tzinfo=sm.get_tzinfo())
        sm.set_spontaneous_next_fire_time(dt)
        assert sm.get_spontaneous_next_fire_time() == dt

        # Reload from disk and verify persistence across restarts
        sm2 = StateManager(sf)
        sm2.load()
        restored = sm2.get_spontaneous_next_fire_time()
        assert restored is not None
        assert restored == dt

        # Clear it
        sm2.set_spontaneous_next_fire_time(None)
        assert sm2.get_spontaneous_next_fire_time() is None

        sm3 = StateManager(sf)
        sm3.load()
        assert sm3.get_spontaneous_next_fire_time() is None


def test_state_scheduled_replies_crud_and_persistence():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()

        # Initially empty
        assert sm.get_scheduled_replies() == []
        assert sm.format_scheduled_replies_context() == "None"

        # Add a oneshot reply
        entry1 = {
            "id": "sched_1001",
            "type": "oneshot",
            "chat_id": -100123,
            "target_msg_id": 42,
            "target_time": "2026-09-27T09:00:00+00:00",
            "interval_str": None,
            "interval_spec": None,
            "description": "Remind Alice about report",
            "created_at": "2026-09-26T12:00:00+00:00",
        }
        s_id1 = sm.add_scheduled_reply(entry1)
        assert s_id1 == "sched_1001"
        assert len(sm.get_scheduled_replies()) == 1
        assert sm.get_scheduled_reply("sched_1001")["description"] == "Remind Alice about report"

        # Add a periodic reply without id (should auto-generate)
        entry2 = {
            "type": "periodic",
            "chat_id": -100123,
            "target_msg_id": None,
            "target_time": "2026-10-01T10:00:00+00:00",
            "interval_str": "1 month",
            "interval_spec": {"years": 0, "months": 1, "days": 0, "hours": 0, "minutes": 0, "seconds": 0},
            "description": "Monthly retro",
            "created_at": "2026-09-26T12:00:00+00:00",
        }
        s_id2 = sm.add_scheduled_reply(entry2)
        assert s_id2.startswith("sched_")
        assert len(sm.get_scheduled_replies()) == 2

        # Verify formatting
        context = sm.format_scheduled_replies_context()
        assert "sched_1001" in context
        assert "Type: oneshot" in context
        assert "Due: 2026-09-27T09:00:00+00:00" in context
        assert "Remind Alice about report" in context
        assert s_id2 in context
        assert "Type: periodic" in context
        assert "Interval: 1 month" in context
        assert "Monthly retro" in context

        # Update
        updated = sm.update_scheduled_reply("sched_1001", {"target_time": "2026-09-27T10:00:00+00:00"})
        assert updated is True
        assert sm.get_scheduled_reply("sched_1001")["target_time"] == "2026-09-27T10:00:00+00:00"
        assert sm.update_scheduled_reply("nonexistent", {"target_time": "2026-09-27T10:00:00+00:00"}) is False

        # Reload from disk and verify persistence
        sm2 = StateManager(sf)
        sm2.load()
        assert len(sm2.get_scheduled_replies()) == 2
        assert sm2.get_scheduled_reply("sched_1001")["target_time"] == "2026-09-27T10:00:00+00:00"

        # Remove
        removed = sm2.remove_scheduled_reply("sched_1001")
        assert removed is True
        assert len(sm2.get_scheduled_replies()) == 1
        assert sm2.get_scheduled_reply("sched_1001") is None
        assert sm2.remove_scheduled_reply("sched_1001") is False

        # Reload again
        sm3 = StateManager(sf)
        sm3.load()
        assert len(sm3.get_scheduled_replies()) == 1
        assert sm3.get_scheduled_reply(s_id2) is not None


def test_state_scheduled_replies_migration():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        # Write state JSON without scheduled_replies key
        with open(sf, "w", encoding="utf-8") as f:
            import json
            json.dump({"version": 1, "language": "English"}, f)

        sm = StateManager(sf)
        sm.load()
        assert sm.get_scheduled_replies() == []
        assert sm.format_scheduled_replies_context() == "None"
def test_state_separated_files_persistence():
    import json
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        sm = StateManager(sf)
        sm.load()

        chf = os.path.join(td, "pletykas-chathistory.json")
        memf = os.path.join(td, "pletykas-memhistory.json")
        pf = os.path.join(td, "pletykas-sysprompt.txt")
        scf = os.path.join(td, "pletykas-sched.json")

        assert os.path.exists(chf)
        assert os.path.exists(memf)
        assert os.path.exists(pf)
        assert os.path.exists(scf)

        # Mutate chat history
        sm.append_chat_message({"id": 100, "text": "hello"})
        with open(chf, "r", encoding="utf-8") as f:
            ch_data = json.load(f)
        assert len(ch_data) == 1
        assert ch_data[0]["id"] == 100

        # Mutate memory history
        sm.append_memory_message({"id": 200, "text": "mem hello"})
        with open(memf, "r", encoding="utf-8") as f:
            mem_data = json.load(f)
        assert len(mem_data) == 1
        assert mem_data[0]["id"] == 200

        # Mutate system prompt
        sm.set_system_prompt("Custom test prompt persona")
        with open(pf, "r", encoding="utf-8") as f:
            p_data = f.read()
        assert p_data == "Custom test prompt persona"

        # Mutate scheduled replies
        s_id = sm.add_scheduled_reply({"type": "oneshot", "target_time": "2026-09-27T12:00:00Z", "description": "task 1"})
        with open(scf, "r", encoding="utf-8") as f:
            sc_data = json.load(f)
        assert len(sc_data) == 1
        assert sc_data[0]["id"] == s_id

        # Verify state.json does NOT contain chat_history, memory_history, system_prompt, scheduled_replies
        with open(sf, "r", encoding="utf-8") as f:
            state_data = json.load(f)
        assert "chat_history" not in state_data
        assert "memory_history" not in state_data
        assert "system_prompt" not in state_data
        assert "scheduled_replies" not in state_data

        # Reload in a new StateManager instance
        sm2 = StateManager(sf)
        sm2.load()
        assert len(sm2.get_chat_history()) == 1
        assert sm2.get_chat_history()[0]["id"] == 100
        assert len(sm2.get_memory_history()) == 1
        assert sm2.get_memory_history()[0]["id"] == 200
        assert sm2.get_system_prompt() == "Custom test prompt persona"
        assert len(sm2.get_scheduled_replies()) == 1
        assert sm2.get_scheduled_replies()[0]["id"] == s_id


def test_state_migration_from_monolithic_state():
    import json
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        chf = os.path.join(td, "pletykas-chathistory.json")
        memf = os.path.join(td, "pletykas-memhistory.json")
        pf = os.path.join(td, "pletykas-sysprompt.txt")
        scf = os.path.join(td, "pletykas-sched.json")

        # Write legacy monolithic state.json
        legacy_data = {
            "version": 1,
            "group_chat_id": -100123,
            "language": "English",
            "chat_history": [{"id": 1, "text": "legacy msg"}],
            "memory_history": [{"id": 10, "text": "legacy mem"}],
            "system_prompt": "Legacy prompt persona",
            "scheduled_replies": [{"id": "sched_legacy", "type": "oneshot", "description": "legacy task"}],
        }
        with open(sf, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        assert not os.path.exists(chf)
        assert not os.path.exists(memf)
        assert not os.path.exists(pf)
        assert not os.path.exists(scf)

        sm = StateManager(sf)
        sm.load()

        # Check in-memory values
        assert len(sm.get_chat_history()) == 1
        assert sm.get_chat_history()[0]["text"] == "legacy msg"
        assert len(sm.get_memory_history()) == 1
        assert sm.get_memory_history()[0]["text"] == "legacy mem"
        assert sm.get_system_prompt() == "Legacy prompt persona"
        assert len(sm.get_scheduled_replies()) == 1
        assert sm.get_scheduled_replies()[0]["id"] == "sched_legacy"

        # Check auxiliary files were written
        assert os.path.exists(chf)
        assert os.path.exists(memf)
        assert os.path.exists(pf)
        assert os.path.exists(scf)

        with open(chf, "r", encoding="utf-8") as f:
            assert json.load(f)[0]["id"] == 1
        with open(memf, "r", encoding="utf-8") as f:
            assert json.load(f)[0]["id"] == 10
        with open(pf, "r", encoding="utf-8") as f:
            assert f.read() == "Legacy prompt persona"
        with open(scf, "r", encoding="utf-8") as f:
            assert json.load(f)[0]["id"] == "sched_legacy"

        # Check state.json was cleaned
        with open(sf, "r", encoding="utf-8") as f:
            cleaned_state = json.load(f)
        assert "chat_history" not in cleaned_state
        assert "memory_history" not in cleaned_state
        assert "system_prompt" not in cleaned_state
        assert "scheduled_replies" not in cleaned_state


def test_state_legacy_history_file_migration():
    import json
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        old_hf = os.path.join(td, "pletykas-history.json")
        new_chf = os.path.join(td, "pletykas-chathistory.json")

        # Write old pletykas-history.json
        with open(old_hf, "w", encoding="utf-8") as f:
            json.dump([{"id": 42, "text": "from old history file"}], f)

        sm = StateManager(sf)
        sm.load()

        assert len(sm.get_chat_history()) == 1
        assert sm.get_chat_history()[0]["id"] == 42
        assert os.path.exists(new_chf)
        assert not os.path.exists(old_hf)


def test_state_separated_files_error_handling():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "state.json")
        chf = os.path.join(td, "pletykas-chathistory.json")
        memf = os.path.join(td, "pletykas-memhistory.json")
        pf = os.path.join(td, "pletykas-sysprompt.txt")
        scf = os.path.join(td, "pletykas-sched.json")

        # Write corrupted files
        with open(sf, "w", encoding="utf-8") as f:
            f.write('{"version": 1}')
        with open(chf, "r" if False else "w", encoding="utf-8") as f:
            f.write("corrupted json {[[")
        with open(memf, "w", encoding="utf-8") as f:
            f.write("not a json")
        with open(pf, "w", encoding="utf-8") as f:
            f.write("   \n  ")  # empty
        with open(scf, "w", encoding="utf-8") as f:
            f.write('{"not": "a list"}')

        sm = StateManager(sf)
        sm.load()

        assert sm.get_chat_history() == []
        assert sm.get_memory_history() == []
        from state import DEFAULT_SYSTEM_PROMPT
        assert sm.get_system_prompt() == DEFAULT_SYSTEM_PROMPT
        assert sm.get_scheduled_replies() == []


def test_state_custom_file_paths():
    with tempfile.TemporaryDirectory() as td:
        sf = os.path.join(td, "my-state.json")
        chf = os.path.join(td, "custom-ch.json")
        memf = os.path.join(td, "custom-mem.json")
        pf = os.path.join(td, "custom-p.txt")
        scf = os.path.join(td, "custom-s.json")

        sm = StateManager(
            file_path=sf,
            chathistory_file_path=chf,
            memhistory_file_path=memf,
            sysprompt_file_path=pf,
            sched_file_path=scf,
        )
        sm.load()

        assert sm.chathistory_file_path == os.path.abspath(chf)
        assert sm.memhistory_file_path == os.path.abspath(memf)
        assert sm.sysprompt_file_path == os.path.abspath(pf)
        assert sm.sched_file_path == os.path.abspath(scf)

        assert os.path.exists(chf)
        assert os.path.exists(memf)
        assert os.path.exists(pf)
        assert os.path.exists(scf)
