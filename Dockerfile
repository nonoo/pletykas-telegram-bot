FROM docker.io/library/python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py /app/

ENV BOT_TOKEN= GROUP_CHAT_ID= ADMIN_USERIDS= \
	STATE_FILE=/app/pletykas-state.json \
	CHATHISTORY_FILE=/app/pletykas-chathistory.json \
	MEMHISTORY_FILE=/app/pletykas-memhistory.json \
	SYSPROMPT_FILE=/app/pletykas-sysprompt.txt \
	SCHED_FILE=/app/pletykas-sched.json \
	MEMORY_FILE=/app/pletykas-memory.json \
	DEEPMEMORY_FILE=/app/pletykas-deepmemory.json \
	MODEL_NAME= MODEL_API_KEY= MODEL_API_BASE= MODEL_THINKING_LEVEL= \
	MODEL_LARGE_NAME= MODEL_LARGE_API_KEY= MODEL_LARGE_API_BASE= MODEL_LARGE_THINKING_LEVEL= \
	MODEL_IMAGE_NAME= MODEL_IMAGE_API_KEY= MODEL_IMAGE_API_BASE= \
	MODEL_IMAGE_SIZE=1K MODEL_IMAGE_THINKING_LEVEL= \
	MODEL_EMBED_NAME=google/gemini-embedding-2 MODEL_EMBED_API_KEY= MODEL_EMBED_API_BASE= MODEL_EMBED_DIM=0

ENTRYPOINT ["python3", "main.py"]
