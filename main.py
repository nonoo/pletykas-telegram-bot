import asyncio
import logging
import sys
import time
from typing import Optional

from telegram.ext import Application, ApplicationBuilder, ContextTypes

from handlers import BotHandlers
from llm import LLMClient
from memory import MemoryManager
from params import Params
from state import StateManager


class HTTPRequestLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "HTTP Request:" in msg or "getUpdates" in msg:
            return False
        return True


class DebounceJobFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "debounce" not in record.getMessage()

def format_duration(seconds: float) -> str:
    secs = int(round(seconds))
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    rem_secs = secs % 60
    if mins < 60:
        return f"{mins}m{rem_secs}s"
    hours = mins // 60
    rem_mins = mins % 60
    return f"{hours}h{rem_mins}m{rem_secs}s"


class APSchedulerJobDurationFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.job_start_times = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if "debounce" in record.getMessage():
            return False
        if record.args and len(record.args) >= 1:
            if "Running job" in record.msg:
                self.job_start_times[str(record.args[0])] = time.monotonic()
            elif "executed successfully" in record.msg and ", took " not in record.msg:
                job_key = str(record.args[0])
                start = self.job_start_times.pop(job_key, None)
                if start is not None:
                    duration = format_duration(time.monotonic() - start)
                    record.msg = f"{record.msg}, took {duration}"
        return True


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

http_request_filter = HTTPRequestLogFilter()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx").addFilter(http_request_filter)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore").addFilter(http_request_filter)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram").addFilter(http_request_filter)
for h in logging.root.handlers:
    h.addFilter(http_request_filter)

job_duration_filter = APSchedulerJobDurationFilter()
logging.getLogger("apscheduler.executors.default").addFilter(job_duration_filter)
logging.getLogger("apscheduler").addFilter(job_duration_filter)

debounce_job_filter = DebounceJobFilter()
logging.getLogger("apscheduler.executors.default").addFilter(debounce_job_filter)
logging.getLogger("apscheduler").addFilter(debounce_job_filter)
logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
logging.getLogger("apscheduler.scheduler").addFilter(debounce_job_filter)
for h in logging.root.handlers:
    h.addFilter(debounce_job_filter)

logger = logging.getLogger("pletykas")


async def notify_admins_startup(app: Application, params: Params) -> None:
    for admin_id in params.admin_user_ids:
        try:
            await app.bot.send_message(chat_id=admin_id, text="🚀 Pletykas bot started")
            logger.info("Sent startup notification to admin %d", admin_id)
        except Exception as e:
            logger.warning("Failed to send startup notification to admin %d: %s", admin_id, e)


def main():
    params = Params()
    try:
        params.parse()
    except Exception as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    state = StateManager(params.state_file, group_chat_id=params.group_chat_id)
    state.load()

    # Synchronize group_chat_id between config params and state
    state_chat_id = state.get_group_chat_id()
    if state_chat_id != 0 and params.is_group_authorized(state_chat_id):
        params.group_chat_id = state_chat_id
    elif params.group_chat_id != 0:
        state.set_group_chat_id(params.group_chat_id)
    if state.is_debug_mode():
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode is active (state.debug=True).")

    memory = MemoryManager(params.memory_file)
    memory.load()

    llm = LLMClient(params, state)
    handlers = BotHandlers(params, state, memory, llm)

    async def post_init(application: Application) -> None:
        await notify_admins_startup(application, params)

        # Schedule spontaneous message timer if enabled
        class DummyContext:
            def __init__(self, app):
                self.application = app
                self.job_queue = app.job_queue
                self.bot = app.bot

        handlers.schedule_spontaneous_job(DummyContext(application))
        handlers.load_and_schedule_pending_replies(DummyContext(application))
        logger.info("Pletykas bot initialized successfully.")

    async def post_shutdown(application: Application) -> None:
        logger.info("Shutting down Pletykas bot...")
        try:
            state.save()
            memory.save()
            await llm.close()
            logger.info("Persisted state and memory files.")
        except Exception as e:
            logger.error("Error during shutdown cleanup: %s", e)
    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Exception while handling an update: %s", context.error, exc_info=context.error)


    app = (
        ApplicationBuilder()
        .token(params.bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    handlers.register(app)
    app.add_error_handler(error_handler)

    logger.info("Starting Pletykas Telegram bot polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
