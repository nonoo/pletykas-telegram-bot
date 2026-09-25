#!/bin/bash

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

if [ -f config.inc.sh ]; then
	. config.inc.sh
fi

if [ ! -d ".venv" ]; then
	echo "Creating virtual environment in .venv..."
	python3 -m venv .venv
fi

REQ_HASH_FILE=".venv/.requirements.hash"
CURRENT_HASH=$(sha256sum requirements.txt 2>/dev/null || true)
STORED_HASH=$(cat "$REQ_HASH_FILE" 2>/dev/null || true)

if [ "$CURRENT_HASH" != "$STORED_HASH" ]; then
	echo "Installing/updating dependencies in .venv..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	echo "$CURRENT_HASH" > "$REQ_HASH_FILE"
fi

BOT_TOKEN="$BOT_TOKEN" \
GROUP_CHAT_ID="$GROUP_CHAT_ID" \
ADMIN_USERIDS="$ADMIN_USERIDS" \
STATE_FILE="${STATE_FILE:-pletykas-state.json}" \
MEMORY_FILE="${MEMORY_FILE:-pletykas-memory.json}" \
MODEL_NAME="${MODEL_NAME}" \
MODEL_API_KEY="${MODEL_API_KEY}" \
MODEL_API_BASE="${MODEL_API_BASE}" \
MODEL_THINKING_LEVEL="$MODEL_THINKING_LEVEL" \
MODEL_LARGE_NAME="${MODEL_LARGE_NAME}" \
MODEL_LARGE_API_KEY="${MODEL_LARGE_API_KEY}" \
MODEL_LARGE_API_BASE="$MODEL_LARGE_API_BASE" \
MODEL_LARGE_THINKING_LEVEL="$MODEL_LARGE_THINKING_LEVEL" \
MODEL_IMAGE_NAME="${MODEL_IMAGE_NAME}" \
MODEL_IMAGE_API_KEY="${MODEL_IMAGE_API_KEY}" \
MODEL_IMAGE_API_BASE="$MODEL_IMAGE_API_BASE" \
MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE}" \
MODEL_IMAGE_THINKING_LEVEL="$MODEL_IMAGE_THINKING_LEVEL" \
exec .venv/bin/python3 main.py "$@"
