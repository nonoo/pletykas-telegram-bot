# Scheduled Replies via LLM Plan

## Context
Add support for the LLM to autonomously schedule replies when requested by users in the group chat. When a user message asks the bot to reply, remind, or perform a check at a later time (one-shot) or on a recurring basis (periodic), the LLM outputs a dedicated schedule block (`<SCHEDULE:oneshot>` or `<SCHEDULE:periodic>`). The bot extracts this block, confirms the schedule conversationally to the user in the current turn, stores the schedule persistently in `pletykas-state.json`, and registers a timer with python-telegram-bot's `JobQueue`. When the timer fires, the bot invokes the LLM with the comprehensive task description provided during scheduling and dispatches the generated reply to the Telegram group chat.

Intervals of arbitrary scale—including multiple days, weeks, months, or years—are supported with calendar accuracy via `dateutil.relativedelta`. To ensure the LLM has complete awareness of scheduled tasks, active timers are injected into LLM prompt contexts for all conversational evaluations and spontaneous messages, allowing the model to answer queries about pending tasks, avoid duplicate timers, and cancel existing timers via `<SCHEDULE:cancel:id>`.

---

## Approach

### 1. Tag Contract & Prompt Instructions (`llm.py`)
- **Block Formats**:
  - **One-shot**:
    ```text
    <SCHEDULE:oneshot>
    Time: <ISO datetime 'YYYY-MM-DD HH:MM:SS', time 'HH:MM', or relative offset '+2h', 'in 30m', 'in 3 days', 'in 2 months', 'in 1 year'>
    Description: <Comprehensive instruction detailing what to say, who requested it, and necessary context>
    </SCHEDULE:oneshot>
    ```
  - **Periodic**:
    ```text
    <SCHEDULE:periodic>
    Interval: <recurrence interval, e.g. '30m', '12h', '1d', '3 days', '2 weeks', '1 month', '3 months', '1 year'>
    Start: <optional first execution time 'YYYY-MM-DD HH:MM:SS' or relative offset>
    Description: <Comprehensive instruction detailing what to say or check periodically>
    </SCHEDULE:periodic>
    ```
  - **Cancellation**:
    ```text
    <SCHEDULE:cancel:sched_id>
    ```
    or `<SCHEDULE:cancel>sched_id</SCHEDULE:cancel>`.
- **Tag Extraction**:
  - Implement `LLMClient._extract_schedule(text: str) -> Tuple[str, Optional[Dict[str, Any]]]`.
  - Extracts `<SCHEDULE:(oneshot|periodic)>([\s\S]*?)(?:</SCHEDULE:(?:oneshot|periodic)>|</SCHEDULE>|$)`.
  - Also extracts `<SCHEDULE:cancel(?::\s*([^>]+))?>` and strips it from text.
  - Parses key-value lines (`Time:`, `Interval:`, `Start:`, `Description:`) from the body.
  - Returns the cleaned user-facing message text (stripped of schedule blocks) and normalized `schedule_spec` dict:
    - Creation: `{"action": "create", "type": "oneshot"|"periodic", "time": str, "interval": str, "start": str, "description": str}`
    - Cancellation: `{"action": "cancel", "schedule_id": str}`.
- **Prompt Context Injection (When the LLM receives scheduled timers)**:
  - **In Conversational Evaluation (`evaluate_and_reply`, `describe_and_reply_image`)**:
    Active scheduled timers are **ALWAYS injected in every evaluation request** under `[Active Scheduled Reminders & Tasks]`.
    Format:
    ```text
    [Active Scheduled Reminders & Tasks]
    - ID: sched_1727339000_1042 | Type: oneshot | Due: 2026-09-27 09:00:00 | Description: Remind Alice about report
    - ID: sched_1727345000_2001 | Type: periodic | Next: 2026-10-01 10:00:00 | Interval: 1 month | Description: Monthly budget review
    ```
    If empty: `None`.
    **Why in every request?**
    1. **Visibility**: Enables the LLM to accurately answer user questions about upcoming reminders ("Mikor szólsz legközelebb?", "What reminders are active?").
    2. **De-duplication**: Prevents creating duplicate timers if someone asks to set a reminder that already exists.
    3. **Cancellation**: Enables the LLM to cancel an existing reminder if a user asks ("Pletyi, mégse szólj holnap!" -> outputs `<SCHEDULE:cancel:sched_1727339000_1042>`).
  - **In `generate_scheduled_reply`**:
    Receives its own specific `scheduled_description`, the schedule type, and the list of other active schedules.
  - **In `generate_spontaneous_message`**:
    Receives the active scheduled list so spontaneous banter doesn't duplicate or collide with a topic scheduled to fire shortly.
  - **In `curate_memory`**:
    **NEVER** injected. Memory curation processes only user and bot conversational history for long-term facts, dynamics, and jokes.
- **System Instructions**:
  - Update `_build_evaluation_instructions(is_direct_trigger, talkativeness, can_search)` to include a `[Scheduled Replies & Reminders]` section explaining the `<SCHEDULE:...>` tags and `<SCHEDULE:cancel:...>`.
  - Instruct the model to write an in-character confirmation to the user in the same turn.
- **LLM Output Signatures**:
  - Update `evaluate_and_reply(...)` return type to `Tuple[Optional[str], Optional[Tuple[str, Optional[int]]], Optional[Dict[str, str]], Optional[Dict[str, Any]]]`.
  - Update `describe_and_reply_image(...)` return type to `Tuple[Optional[str], Optional[Tuple[str, Optional[int]]], Optional[Dict[str, str]], str, Optional[Dict[str, Any]]]`.
  - Update callsites and tests to accept the updated tuples.
- **Dedicated Scheduled Reply Generation**:
  - Implement `LLMClient.generate_scheduled_reply(...)`:
    ```python
    async def generate_scheduled_reply(
        self,
        system_prompt: str,
        memory_context: str,
        transcript: str,
        scheduled_description: str,
        scheduled_type: str,
        timezone_str: str,
    ) -> Optional[str]
    ```
  - Uses system prompt, memory, transcript, current time, and `[Scheduled Task Execution]` with `scheduled_description`.
  - Instructs model to speak naturally in character without mentioning timers or AI automation.

---

### 2. Multi-Day, Multi-Month & Multi-Year Intervals (`handlers.py`)
- **Calendar-Accurate Interval Representation**:
  - Use `from dateutil.relativedelta import relativedelta`.
  - Implement `parse_interval_relativedelta(interval_str: str) -> Optional[relativedelta]`:
    - Supports regex matching for quantity + unit:
      - **Years**: `y`, `yr`, `yrs`, `year`, `years` -> `relativedelta(years=N)`
      - **Months**: `mo`, `month`, `months` -> `relativedelta(months=N)`
      - **Weeks**: `w`, `wk`, `wks`, `week`, `weeks` -> `relativedelta(weeks=N)`
      - **Days**: `d`, `day`, `days` -> `relativedelta(days=N)`
      - **Hours**: `h`, `hr`, `hrs`, `hour`, `hours` -> `relativedelta(hours=N)`
      - **Minutes**: `m`, `min`, `mins`, `minute`, `minutes` -> `relativedelta(minutes=N)`
      - **Seconds**: `s`, `sec`, `secs`, `second`, `seconds` -> `relativedelta(seconds=N)`
    - Also parses compound intervals if given (e.g. `1 month 2 days`).
    - Clamps minimum interval to 60 seconds.
  - Implement `serialize_interval(rd: relativedelta) -> Dict[str, int]` and `deserialize_interval(d: Dict[str, int]) -> relativedelta` for JSON state persistence.
  - Stored in state entry as `interval_spec`: `{"years": 0, "months": 1, "days": 0, "hours": 0, "minutes": 0, "seconds": 0}` and human-readable `interval_str`: `"1 month"`.
  - Advancing periodic schedule:
    `next_dt = target_dt + rd`
    Preserves exact calendar days and times across months with varying lengths (28, 30, 31 days) and leap years without drift.
- **Date Parsing with Relativedelta**:
  - `parse_schedule_time(time_str: str, tz: ZoneInfo, now: datetime) -> Optional[datetime]`:
    - Handles relative offsets with relativedelta: `in 3 days`, `+2 months`, `in 1 year`, `+2h`, `in 30m`.
    - Handles time-only format: `HH:MM` or `HH:MM:SS` (advances by 1 day if time today already passed).
    - Handles absolute timestamps via `dateutil.parser.parse(time_str)`. If naive, attaches `tz`; if aware, converts to `tz`.

---

### 3. Persistent State Management (`state.py`)
- **State Schema**:
  - Add `"scheduled_replies": []` to `DEFAULT_STATE` in `state.py`.
  - Entry schema:
    ```json
    {
      "id": "sched_1727339000_1042",
      "type": "oneshot",
      "chat_id": -1001234567890,
      "target_msg_id": 1042,
      "target_time": "2026-10-01T10:00:00+02:00",
      "interval_str": "1 month",
      "interval_spec": {"years": 0, "months": 1, "days": 0, "hours": 0, "minutes": 0, "seconds": 0},
      "description": "Monthly team retrospective reminder.",
      "created_at": "2026-09-26T12:00:00+02:00"
    }
    ```
- **StateManager Methods**:
  - `get_scheduled_replies() -> List[Dict[str, Any]]`: Returns defensive copy.
  - `add_scheduled_reply(entry: Dict[str, Any]) -> str`: Saves and returns ID.
  - `remove_scheduled_reply(schedule_id: str) -> bool`: Removes by ID and saves.
  - `get_scheduled_reply(schedule_id: str) -> Optional[Dict[str, Any]]`: Retrieves entry by ID.
  - `update_scheduled_reply(schedule_id: str, updates: Dict[str, Any]) -> bool`: Updates fields and saves.
  - `format_scheduled_replies_context() -> str`: Returns formatted summary string of all active timers for injection into LLM prompts.

---

### 4. Job Scheduling, Execution & Cancellation (`handlers.py` & `main.py`)
- **Registration on Evaluation**:
  - In `_execute_evaluation`:
    - If `schedule_spec["action"] == "create"`:
      - Parse `time_str` (or `start`), `interval_str`, and `description`.
      - Compute `target_dt` and `relativedelta`.
      - Generate ID: `f"sched_{int(time.time())}_{trigger_msg_id}"`.
      - Save to `state.add_scheduled_reply(entry)`.
      - Call `self.schedule_reply_job(context, entry)`.
    - If `schedule_spec["action"] == "cancel"`:
      - Target ID = `schedule_spec["schedule_id"]`.
      - Cancel in `job_queue`: `self.cancel_reply_job(context, target_id)`.
      - Remove from state: `self.state.remove_scheduled_reply(target_id)`.
- **`schedule_reply_job(context, entry: Dict[str, Any]) -> None`**:
  - Computes `delay_sec = max(1.0, (target_dt - now).total_seconds())`.
  - Calls `context.job_queue.run_once(self._scheduled_reply_callback, when=delay_sec, name=f"scheduled_reply_{entry['id']}", data={"schedule_id": entry["id"]})`.
- **Callback `_scheduled_reply_callback(context: ContextTypes.DEFAULT_TYPE)`**:
  - Retrieves `schedule_id` from `job.data`.
  - Retrieves entry from `self.state.get_scheduled_reply(schedule_id)`. If missing, returns.
  - Calls `self.llm.generate_scheduled_reply(...)` with `description`, `type`, memory, and recent transcript.
  - Dispatches resulting text to `entry["chat_id"]` (attempting `reply_to_message_id=entry.get("target_msg_id")`, falling back to chat root if original message deleted).
  - Records message in `chat_history` and `memory_history`.
  - If `type == "oneshot"`: Removes entry via `self.state.remove_scheduled_reply(schedule_id)`.
  - If `type == "periodic"`:
    - Computes `next_dt = target_dt + deserialize_interval(entry["interval_spec"])`.
    - If `next_dt <= now`: advances `next_dt` repeatedly until it is in the future.
    - Updates `target_time` in state: `self.state.update_scheduled_reply(schedule_id, {"target_time": next_dt.isoformat()})`.
    - Re-schedules via `self.schedule_reply_job(context, entry)`.
- **Application Startup Resumption (`main.py` & `handlers.py`)**:
  - `handlers.load_and_schedule_pending_replies(context: ContextTypes.DEFAULT_TYPE) -> None`:
    - Called in `main.py` `post_init` via `DummyContext(application)`.
    - Loads all pending timers directly from `self.state.get_scheduled_replies()` (which was loaded from `pletykas-state.json` on app startup).
    - For **oneshot** timers:
      - If `target_dt > now`: logs resumption and registers a `run_once` job in `context.job_queue` with `delay = (target_dt - now).total_seconds()`.
      - If `target_dt <= now` and elapsed time is < 15 minutes (e.g. quick app restart or update): fires immediately by running `_scheduled_reply_callback` via a 1-second `run_once` job, ensuring the user's reminder isn't lost.
      - If `target_dt <= now` and elapsed time is >= 15 minutes: logs a warning that the timer expired while the bot was offline, removes it from `state`, and skips execution.
    - For **periodic** timers:
      - If `target_dt > now`: registers a `run_once` job for `(target_dt - now).total_seconds()`.
      - If `target_dt <= now`: advances `target_dt` using `deserialize_interval(entry["interval_spec"])` until it reaches the next future execution time, updates `target_time` in `pletykas-state.json`, and schedules the timer.
    - Logs summary count: `logger.info("Loaded %d scheduled replies from state file (%d active).", total, active_count)`.
  - In `main.py` `post_init`: `handlers.load_and_schedule_pending_replies(DummyContext(application))` is called right after `handlers.schedule_spontaneous_job(...)`.

---

### 5. Administration & Inspection (`handlers.py`)
- **Status Display (`/status`)**:
  - Include active scheduled replies counter and next upcoming run time:
    `• Scheduled Replies: N active (Next: <timestamp> | None)`
- **Admin Command `/scheduled`**:
  - `/scheduled` or `/scheduled list`: Lists active scheduled replies with ID, type, interval, next run time, and description.
  - `/scheduled cancel <id>`: Manually cancels and deletes a scheduled reply.
  - Register `/scheduled` in `register()` and add to `HELP_MESSAGE`.

---

### 6. Documentation (`AGENTS.md`, `README.md`)
- Document `<SCHEDULE:oneshot>`, `<SCHEDULE:periodic>`, and `<SCHEDULE:cancel:id>` in `AGENTS.md` Section 4.
- Document `scheduled_replies` in `AGENTS.md` persistent state schema.
- Document `/scheduled` in `README.md` admin command table.

## Critical Files & Anchors

1. `llm.py`:
   - `_extract_schedule`: Add schedule block regex extraction and key parsing.
   - `_build_evaluation_instructions`: Inject `[Scheduled Replies & Reminders]` prompt guidance.
   - `evaluate_and_reply`: Return `(reply_text, reaction, image_spec, schedule_spec)`.
   - `generate_scheduled_reply`: New LLM prompt method for executing scheduled tasks.
2. `state.py`:
   - `DEFAULT_STATE`: Add `"scheduled_replies": []`.
   - Methods: `get_scheduled_replies`, `add_scheduled_reply`, `remove_scheduled_reply`, `get_scheduled_reply`, `update_scheduled_reply`.
3. `handlers.py`:
   - `parse_schedule_time` & `parse_interval_relativedelta`: Datetime and duration parsing helpers.
   - `_execute_evaluation`: Handle `schedule_spec`, persist to state, and call `schedule_reply_job`.
   - `schedule_reply_job`, `_scheduled_reply_callback`, `load_and_schedule_pending_replies`: Job lifecycle management.
   - `cmd_status` & `cmd_scheduled`: Status reporting and cancellation.
4. `main.py`:
   - `post_init`: Add `handlers.load_and_schedule_pending_replies(DummyContext(application))`.
5. `tests/`:
   - `tests/test_llm.py`: Tests for tag extraction, prompt building, and scheduled reply generation.
   - `tests/test_state.py`: Tests for CRUD operations on `scheduled_replies`.
   - `tests/test_handlers.py`: Tests for schedule job registration, callback execution, restart recovery, and `/scheduled` command.

---

## Verification

1. **Unit Tests (`tests/test_llm.py`)**:
   - Verify `_extract_schedule` correctly extracts `<SCHEDULE:oneshot>` with `Time:` and `Description:`, stripping the block and preserving user confirmation text.
   - Verify `_extract_schedule` correctly extracts `<SCHEDULE:periodic>` with `Interval:`, `Start:`, and `Description:`.
   - Verify `evaluate_and_reply` returns parsed `schedule_spec`.
   - Verify `generate_scheduled_reply` constructs prompt including `scheduled_description` and calls the model.
2. **State Tests (`tests/test_state.py`)**:
   - Verify adding, retrieving, updating, and removing scheduled replies in `StateManager` with atomic file persistence and reload.
3. **Handler & JobQueue Integration Tests (`tests/test_handlers.py`)**:
   - Simulate a user message requesting a scheduled reply: verify LLM confirmation is dispatched, schedule entry is saved to state, and `job_queue.run_once` is scheduled.
   - Simulate `_scheduled_reply_callback`: verify `generate_scheduled_reply` is called, message is sent to chat, and oneshot entry is cleaned up (or periodic entry is rescheduled).
   - Test `load_and_schedule_pending_replies` simulates bot restart and verifies pending future jobs are rescheduled.
   - Test `/scheduled` and `/scheduled cancel <id>` in private admin chat.
4. **Full Suite Execution**:
   - Run `.venv/bin/pytest -v` to ensure all existing and new tests pass.
   - Run `.venv/bin/pytest tests/smoke_test.py -v`.

---

## Assumptions & Contingencies

- **Timezone Anchor**: The LLM interprets times relative to the bot's configured timezone (`state.get_timezone()`), which is already passed into prompts (`Current Time: ...`, `Timezone: ...`).
- **Sleep Schedule during Scheduled Replies**: If a scheduled reply triggers during quiet hours (`is_sleeping()`), oneshot replies (explicitly user-requested for that time) still execute; periodic replies defer or run silently depending on configuration.
- **Unparseable Schedule Timestamp**: If the LLM generates an invalid or past timestamp that cannot be resolved by `parse_schedule_time`, log a warning, do not crash, and fall back to `now + timedelta(seconds=60)` if relative intent was clear or skip scheduling.
