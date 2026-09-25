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
        assert sm.get_cooldown_sec() == 3
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
        sm.set_cooldown_sec(5)
        sm.set_search_grounding_active(True)
        sm.set_debug_mode(True)
        sm.set_nicknames(["pletyka", "@pleti", "pletyka"])

        sm2 = StateManager(sf)
        sm2.load()
        assert sm2.get_language() == "English"
        assert sm2.get_talkativeness() == 9
        assert sm2.get_cooldown_sec() == 5
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

