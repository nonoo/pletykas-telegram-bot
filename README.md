# Pletykas Telegram Bot

<p align="center">
  <img src="logo.png" alt="Pletykas Telegram Bot" width="250">
</p>

An autonomous, multi-modal Telegram group chatbot designed to seamlessly integrate into group discussions as a witty, observant, and subtly gossipy participant.

## Key Features

- **Single Group Restriction**: Locks exclusively to an authorized `GROUP_CHAT_ID`. If added to unauthorized chats, it logs an alert and leaves automatically.
- **Strict Private Admin Interface**: Administrative slash commands are only accepted in private chats with authorized `ADMIN_USERIDS`.
- **Pure LLM Autonomy**: Decides dynamically whether to reply, react with an emoji, request image generation/editing, or remain silent (`<NO_REPLY>`).
- **No Consecutive Duplicates**: A text reply that repeats the bot's own previous group message (identical or at least 90% similar after normalization) is suppressed before dispatch, for conversational, scheduled, and spontaneous messages alike.
- **Debounced Cooldown Timer**: Rapid consecutive messages reset a debounce timer (`cooldown_sec`), batching human chatter into a cohesive context before model evaluation.
- **Multimodal Image Support**: Ingests photos directly, generates visual descriptions, and supports text-to-image and image-to-image editing using `<GENERATE_IMAGE>`.
- **Dynamic Memory Management**: Automatically distills facts, group dynamics, and inside jokes every 20 messages into `pletykas-memory.json` with rolling `.bak` backups.
- **Two-Tier Memory**: Aging facts are demoted to a deep store (`pletykas-deepmemory.json`); relevant deep entries are recalled automatically by embedding similarity and injected into the prompt only when they clear the relevance threshold.
- **Quiet Hours & Chat Revival**: Optional sleep schedule (`/sleep`) and unprompted conversational revival with polls or banter (`/spontaneous`).
- **Capability Escalation & Retry**: The fast primary model automatically escalates to the large model (`<RETRY_WITH_LARGE_MODEL>`) when real-time web search or capabilities beyond its reach are needed.
- **Autonomous Scheduled Replies & Reminders**: The LLM can autonomously schedule one-shot or periodic reminders/checks (`<SCHEDULE:oneshot>`, `<SCHEDULE:periodic>`). Timers persist across restarts in `pletykas-sched.json` and are viewable/cancellable via `/scheduled`.

---

## Architecture & Models

- **Primary Conversational Model** (`MODEL_NAME`): Default `deepseek-flash` (or Google Gemini/Gemma models).
- **Vision Model** (`MODEL_LARGE_NAME`): Default `gemini-flash-latest` for high-fidelity photo captioning and visual descriptions.
- **Image Generation Model** (`MODEL_IMAGE_NAME`, `MODEL_IMAGE_SIZE`): Default `gemini-3.1-flash-lite-image` (or OpenAI DALL-E / compatible image endpoints), default size `1K`.
- **Embedding Model** (`MODEL_EMBED_NAME`): OpenAI-compatible `/embeddings` route for deep-memory recall, default `google/gemini-embedding-2`. Falls back to the primary key/base unless `MODEL_EMBED_*` is set.

---

## Quick Start

### 1. Configuration

Copy the example configuration file:
```bash
cp config.inc.sh-example config.inc.sh
```

Edit `config.inc.sh` with your credentials:
```bash
BOT_TOKEN="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
GROUP_CHAT_ID="-1001234567890"
ADMIN_USERIDS="12345678,98765432"

MODEL_NAME="deepseek-flash"
MODEL_API_KEY="your-deepseek-or-llm-api-key"
MODEL_API_BASE="https://api.deepseek.com"
MODEL_THINKING_LEVEL="" # Empty instructs model not to think (default); or minimal, low, medium, high, budget

MODEL_LARGE_NAME="gemini-flash-latest"
MODEL_LARGE_API_KEY="your-gemini-api-key"
MODEL_LARGE_API_BASE=""
MODEL_LARGE_THINKING_LEVEL=""

MODEL_IMAGE_NAME="gemini-3.1-flash-lite-image"
MODEL_IMAGE_API_KEY="your-gemini-api-key"
MODEL_IMAGE_API_BASE=""
MODEL_IMAGE_SIZE="1K"
MODEL_IMAGE_THINKING_LEVEL=""

MODEL_EMBED_NAME="google/gemini-embedding-2" # Deep-memory recall embeddings
MODEL_EMBED_API_KEY="" # Empty falls back to MODEL_API_KEY
MODEL_EMBED_API_BASE="" # Empty falls back to MODEL_API_BASE
```

> Deep-memory recall requires the embedding provider to expose an OpenAI-compatible `/embeddings` endpoint (e.g. OpenRouter, or Google's OpenAI-compat layer at `https://generativelanguage.googleapis.com/v1beta/openai` with `MODEL_EMBED_NAME="gemini-embedding-2"`).

### 2. Run Locally

Execute `run.sh` to initialize the virtual environment, install dependencies, and start the bot:
```bash
./run.sh
```

### 3. Containerized Deployment

Build and run using Podman or Docker:
```bash
./01-buildah-image.sh
podman run -d --name pletykas-telegram-bot --env-file config.inc.sh pletykas-telegram-bot:latest
```

---

## Administrative Commands (Private Chat Only)

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `/start`, `/help` | — | Displays formatted command overview |
| `/status` | — | Shows active parameters, uncurated message stats, and model settings |
| `/talkativeness` | `[1-10]` | Views or adjusts autonomous conversation entry frequency |
| `/cooldown` | `[sec]` | Views or adjusts debounce delay (in seconds; 0 disables) |
| `/nicknames` | `[nick1, nick2...]` | Views, sets, or clears comma-separated direct trigger nicknames |
| `/language` | `[lang]` | Views or updates active conversation language (e.g. `English`, `Hungarian`) |
| `/timezone` | `[tz]` | Views or sets the IANA timezone (e.g. `Europe/Budapest`) |
| `/sleep` | `[on\|off] [start] [end]` | Configures quiet hours (e.g. `/sleep on 23:00 07:00`) |
| `/spontaneous` | `[on\|off\|now]` | Enables, disables, or views periodic spontaneous messages |
| `/spontaneous_interval` | `[min] [max]` | Views or adjusts the random timer interval in hours (default: `2 4`) |
| `/spontaneous_now` | — | Instantly generates and dispatches a spontaneous message or poll to the group |
| `/scheduled` | `[list\|cancel <id>]` | Views active scheduled reminders or cancels a specific timer by ID |
| `/prompt` | `[load\|reset]` | Uploads system prompt file as-is, expects a text file upload to replace prompt, or resets to default |
| `/grounding` | `[on\|off]` | Toggles Google Search grounding tool |
| `/search_small` | `[on\|off]` | Toggles whether small model uses search grounding directly (default: `OFF` delegates to large model) |
| `/image_large` | `[on\|off]` | Toggles large model for image interpretation (default: `ON` instant large model; `OFF` tries small model first) |
| `/debug` | `[on\|off]` | Toggles debug streaming to stdout (raw LLM payloads and Telegram group messages) |
| `/memories` | `[load]` | Uploads memories JSON file as-is, or enters waiting mode to validate and load an uploaded JSON file |
| `/deepmemories` | `[load]` | Uploads deep memories JSON file as-is, or enters waiting mode to validate and load an uploaded JSON file |
| `/archive_age` | `[days]` | Views or adjusts the age in days after which hot facts are archived to deep memory (0 disables) |
| `/cancel` | — | Aborts a pending memories, deep memories, or prompt file upload |
| `/curate` | — | Manually triggers immediate LLM reflection and consolidation |

---

## Testing

Run the automated test suite with pytest:
```bash
.venv/bin/pytest -v
```
All unit, integration, and smoke tests will execute against local mocks without requiring live API keys.

## Contributors

- Norbert Varga [nonoo@nonoo.hu](mailto:nonoo@nonoo.hu)

## Donations

If you find this bot useful then [buy me a beer](https://paypal.me/ha2non). :)

## License

[MIT](LICENSE)
