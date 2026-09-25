# AGENTS.md - Developer & Agent Guide for Pletykas Telegram Bot

This document outlines the architecture, design principles, invariants, and implementation details of `pletykas-telegram-bot`.

---

## 1. System Overview & Core Principles

`pletykas-telegram-bot` is an autonomous group-chat assistant designed with the persona of "Pletykas": an observant, witty, and subtly gossipy chat participant.

### Primary Operating Invariants
1. **Single Group Lockdown**: The bot operates exclusively in the designated `GROUP_CHAT_ID`. If added to any other group or supergroup, it immediately calls `context.bot.leave_chat(chat.id)`. Supergroup migrations (`message.migrate_to_chat_id`) update the configured chat ID in memory and state.
2. **Private-Only Administration**: All `/` commands are strictly restricted to private chats with user IDs specified in `ADMIN_USERIDS`. Private messages are never forwarded to the LLM or added to conversation transcripts.
3. **Pure LLM Autonomy**: The model decides when to speak, when to react with an emoji, when to request image generation, and when to remain completely silent using `<NO_REPLY>`.
4. **Debounced Cooldown Batching**: When messages arrive in the group, a debounce timer (`cooldown_sec`) is started or reset. Only when the conversation pauses for `cooldown_sec` seconds does background LLM evaluation execute. Direct triggers (`@mention` or replies to the bot) bypass debounce and evaluate immediately.
5. **No Indicator Leaks**: `ChatAction.TYPING` is never sent. `ChatAction.UPLOAD_PHOTO` is sent strictly during image generation/editing when the model outputs `<GENERATE_IMAGE>`.
6. **Capability Escalation & Retry**: When the primary (small) model lacks capabilities to fulfill a request (e.g. real-time Google Search), it outputs `<RETRY_WITH_LARGE_MODEL>`. The LLM client intercepts this signal and automatically retries using the large model with Google Search grounding active.
7. **Telegram HTML Markdown**: The effective system prompt enforces Telegram HTML markdown restricted strictly to `b`, `i`, `u`, `s`, `a`, `code`, and `blockquote`, while prohibiting other tags and LaTeX. Message dispatches render with `ParseMode.HTML` with automatic fallback to plain text if malformed tags occur.
8. **Image Interpretation Model Selection**: Controlled by the `image_interpretation_large_model` state setting (default: `True`). When `True`, the large vision model is used directly and immediately without attempting the small model. When toggled to `False` (via `/image_large off`), the small model is tried first with resized/compressed images, with automatic fallback to the large model if it fails or signals `<RETRY_WITH_LARGE_MODEL>`.
9. **Debug Mode Streaming**: Controlled by the `debug` state setting (toggled via `/debug [on|off]`). When active, raw LLM request/response payloads, all incoming Telegram group messages (with sender, IDs, media type, and content), and all outgoing Telegram group dispatches (replies, photos, polls, reactions) are printed directly to stdout with structured banners.

---

## 2. Codebase Architecture

```
pletykas-telegram-bot/
├── main.py                     # Entry point, PTB ApplicationBuilder, lifecycle hooks, logging
├── params.py                   # CLI & ENV argument parser, fallback logic, access validation
├── state.py                    # StateManager: atomic persistence of pletykas-state.json, usage stats, sliding histories
├── memory.py                   # MemoryManager: atomic persistence of pletykas-memory.json, schema validation, backup rotation
├── llm.py                      # Multi-model client: Google GenAI + OpenAI-compatible endpoints, tag parsing
├── handlers.py                 # BotHandlers: message ingestion, debounce, dispatch, admin slash commands, spontaneous revival
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── run.sh                      # Shell launcher with virtualenv setup & env loading
├── Dockerfile                  # Container definition
├── 01-buildah-image.sh         # Container build script
├── 02-buildah-login.sh         # Registry login script
├── 03-buildah-push.sh          # Container push script
├── 03-clean.sh                 # Container cleanup script
├── config.inc.sh-example       # Sample environment configuration
├── README.md                   # User and setup guide
└── tests/
    ├── test_params.py          # Unit tests for CLI and environment parsing
    ├── test_state.py           # Unit tests for state persistence, timezone, sleep schedule, rollups
    ├── test_memory.py          # Unit tests for memory CRUD, validation, backup rotation
    ├── test_llm.py             # Unit tests for image compression, tag extraction, memory curation
    ├── test_handlers.py        # Unit tests for message handling, debounce, admin commands
    └── smoke_test.py           # End-to-end integration test
```

---

## 3. Persistent Data Schemas

### State (`pletykas-state.json`)
Managed by `StateManager` via atomic temporary file replacement (`os.replace`):
```json
{
  "version": 1,
  "group_chat_id": -1001234567890,
  "language": "English",
  "timezone": "Europe/Budapest",
  "talkativeness": 5,
  "cooldown_sec": 3,
  "search_grounding": false,
  "debug": false,
  "nicknames": [
    "pletyi",
    "pletyo"
  ],
  "system_prompt": "...",
  "sleep_schedule": {
    "enabled": false,
    "sleep_start": "23:00",
    "sleep_end": "07:00"
  },
  "spontaneous_messages": {
    "enabled": false,
    "min_hours": 2.0,
    "max_hours": 4.0
  },
  "chat_history": [],
  "memory_history": [],
  "messages_since_last_curation": 0
}
```
- `chat_history`: Ring buffer of the last 20 messages injected into prompt transcripts.
- `memory_history`: Ring buffer of the last 30 messages maintained for LLM memory curation.

### Memory (`pletykas-memory.json`)
Managed by `MemoryManager` via atomic replacement:
```json
{
  "version": 1,
  "last_curated_at": "2026-09-25T12:00:00Z",
  "memories": [
    {
      "topic": "Alice",
      "content": "Lives in Berlin and works with Rust.",
      "created_at": "2026-09-25T12:00:00Z"
    }
  ],
  "dynamics": [
    {
      "members": ["Alice", "Bob"],
      "relation": "Longtime collaborators.",
      "created_at": "2026-09-25T12:00:00Z"
    }
  ],
  "inside_jokes": [
    {
      "title": "Tab War",
      "context": "Debate about indentation style.",
      "created_at": "2026-09-25T12:00:00Z"
    }
  ]
}
```
- ID-Free Schema: Memory entries are organized cleanly without artificial identifier tokens (`mem_1`, etc.).
- File Upload Administration: `/memories` uploads the JSON file as-is; `/memories load` validates and applies user-uploaded JSON files. Similarly, `/prompt` uploads the system prompt text file as-is; `/prompt load` validates and applies user-uploaded text files.
- Backups: Rotated automatically before commits, keeping the 20 most recent `.bak` files.

---

## 4. Model Routing & Tag Contract

### Protocols
- If `model_api_base` is empty and the model name begins with `gemini`, `gemma`, or `imagen`, `google-genai` SDK is used asynchronously (`client.aio`).
- Otherwise, requests are routed to OpenAI-compatible `/chat/completions`, `/images/generations`, and `/images/edits`.

### Special Protocol Tags
- `<NO_REPLY>`: The model decides to stay silent.
- `<REACTION:emoji>` or `<REACTION:emoji:message_id>`: Dispatches Telegram reaction to the trigger message or target message ID.
- `<GENERATE_IMAGE>`:
  ```text
  <GENERATE_IMAGE>
  Prompt: <English visual prompt>
  Caption: <In-character message>
  Source: <message ID | reply | new>
  Mode: <generate | modify>
  </GENERATE_IMAGE>
  ```
- `<IMAGE_DESCRIPTION>`: Generated by the multimodal large model when processing incoming photos; stored into history to give subsequent text models context.
- `<POLL>`: Formats an unprompted Telegram poll for spontaneous chat revival.

---

## 5. Testing & Verification

All automated tests use standard `pytest`:
```bash
.venv/bin/pytest -v
```

Testing guidelines:
- Mocks should avoid network I/O.
- Telegram `Update` and `Message` mocks must specify valid attributes (e.g., `migrate_to_chat_id=None`, `date=None`).
- When testing `LLMClient._call_genai`, ensure `params.model_api_base = ""` and a supported model name is set.
