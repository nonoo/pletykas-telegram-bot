import argparse
import logging
import os
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


class Params:
    def __init__(self):
        self.bot_token: str = ""
        self.group_chat_id: int = 0
        self.admin_user_ids: List[int] = []
        self.state_file: str = "pletykas-state.json"
        self.memory_file: str = "pletykas-memory.json"

        # Primary conversational model
        self.model_name: str = ""
        self.model_api_key: str = ""
        self.model_api_base: str = ""
        self.model_thinking_level: str = ""

        # Large Multimodal Model for vision/images
        self.model_large_name: str = ""
        self.model_large_api_key: str = ""
        self.model_large_api_base: str = ""

        self.model_large_thinking_level: str = ""
        # Image generation / editing model
        self.model_image_name: str = ""
        self.model_image_api_key: str = ""
        self.model_image_api_base: str = ""
        self.model_image_size: str = "1K"
        self.model_image_thinking_level: str = ""

    @property
    def effective_large_api_key(self) -> str:
        return self.model_large_api_key or self.model_api_key

    @property
    def effective_large_api_base(self) -> str:
        if self.model_large_api_base:
            return self.model_large_api_base
        if self.model_large_name == self.model_name:
            return self.model_api_base
        return ""
    @property
    def effective_large_thinking_level(self) -> str:
        return self.model_large_thinking_level or self.model_thinking_level


    @property
    def effective_image_api_key(self) -> str:
        return self.model_image_api_key or self.model_api_key

    @property
    def effective_image_api_base(self) -> str:
        if self.model_image_api_base:
            return self.model_image_api_base
        if self.model_image_name == self.model_name:
            return self.model_api_base
        return ""
    @property
    def effective_image_thinking_level(self) -> str:
        return self.model_image_thinking_level or self.model_thinking_level


    def parse(self, args: Optional[List[str]] = None) -> None:
        parser = argparse.ArgumentParser(description="Pletykas Telegram Bot")

        parser.add_argument(
            "--bot-token",
            dest="bot_token",
            default=os.environ.get("BOT_TOKEN", ""),
            help="Telegram Bot Token",
        )
        parser.add_argument(
            "--group-chat-id",
            dest="group_chat_id",
            default=os.environ.get("GROUP_CHAT_ID", "0"),
            help="Authorized Telegram Group Chat ID (e.g. -1001234567890)",
        )
        parser.add_argument(
            "--admin-user-ids",
            dest="admin_user_ids",
            default=os.environ.get("ADMIN_USERIDS", ""),
            help="Comma-separated Telegram user IDs of administrators",
        )
        parser.add_argument(
            "--state-file",
            dest="state_file",
            default=os.environ.get("STATE_FILE", "pletykas-state.json"),
            help="Path to persistent state JSON file",
        )
        parser.add_argument(
            "--memory-file",
            dest="memory_file",
            default=os.environ.get("MEMORY_FILE", "pletykas-memory.json"),
            help="Path to dynamic memory JSON file",
        )

        # Primary model
        parser.add_argument(
            "--model-name",
            dest="model_name",
            default=os.environ.get("MODEL_NAME", ""),
            help="Primary conversational model name",
        )
        parser.add_argument(
            "--model-api-key",
            dest="model_api_key",
            default=os.environ.get("MODEL_API_KEY", os.environ.get("LLM_API_KEY", "")),
            help="API key for primary conversational model",
        )
        parser.add_argument(
            "--model-api-base",
            dest="model_api_base",
            default=os.environ.get("MODEL_API_BASE", ""),
            help="Base URL for primary conversational model API",
        )
        parser.add_argument(
            "--model-thinking-level",
            dest="model_thinking_level",
            default=os.environ.get("MODEL_THINKING_LEVEL", ""),
            help="Thinking level for primary model (e.g. minimal, low, medium, high, 0; empty disables)",
        )

        # Large multimodal model
        parser.add_argument(
            "--model-large-name",
            dest="model_large_name",
            default=os.environ.get("MODEL_LARGE_NAME", ""),
            help="Large Multimodal Model name for vision",
        )
        parser.add_argument(
            "--model-large-api-key",
            dest="model_large_api_key",
            default=os.environ.get("MODEL_LARGE_API_KEY", ""),
            help="API key for large multimodal model (falls back to primary API key)",
        )
        parser.add_argument(
            "--model-large-api-base",
            dest="model_large_api_base",
            default=os.environ.get("MODEL_LARGE_API_BASE", ""),
            help="Base URL for large multimodal model API",
        )
        parser.add_argument(
            "--model-large-thinking-level",
            dest="model_large_thinking_level",
            default=os.environ.get("MODEL_LARGE_THINKING_LEVEL", ""),
            help="Thinking level for vision/large model (e.g. minimal, low, medium, high, 0; empty disables)",
        )

        # Image generation model
        parser.add_argument(
            "--model-image-name",
            dest="model_image_name",
            default=os.environ.get("MODEL_IMAGE_NAME", ""),
            help="Image generation/editing model name",
        )
        parser.add_argument(
            "--model-image-api-key",
            dest="model_image_api_key",
            default=os.environ.get("MODEL_IMAGE_API_KEY", ""),
            help="API key for image model (falls back to primary API key)",
        )
        parser.add_argument(
            "--model-image-api-base",
            dest="model_image_api_base",
            default=os.environ.get("MODEL_IMAGE_API_BASE", ""),
            help="Base URL for image model API",
        )
        parser.add_argument(
            "--model-image-size",
            dest="model_image_size",
            default=os.environ.get("MODEL_IMAGE_SIZE", "1K"),
            help="Image generation output size (default: 1K)",
        )
        parser.add_argument(
            "--model-image-thinking-level",
            dest="model_image_thinking_level",
            default=os.environ.get("MODEL_IMAGE_THINKING_LEVEL", ""),
            help="Thinking level for image model (e.g. minimal, low, medium, high, 0; empty disables)",
        )

        parsed_args = parser.parse_args(args)

        self.bot_token = parsed_args.bot_token.strip()
        if not self.bot_token:
            raise ValueError("bot_token must be non-empty (via --bot-token or BOT_TOKEN)")

        try:
            self.group_chat_id = int(parsed_args.group_chat_id)
        except (ValueError, TypeError):
            raise ValueError(f"group_chat_id must be a valid integer: {parsed_args.group_chat_id}")
        if self.group_chat_id == 0:
            raise ValueError("group_chat_id must be non-zero (via --group-chat-id or GROUP_CHAT_ID)")

        self.admin_user_ids = []
        raw_admins = parsed_args.admin_user_ids
        if raw_admins:
            items = [item.strip() for item in str(raw_admins).split(",") if item.strip()]
            for item in items:
                try:
                    self.admin_user_ids.append(int(item))
                except ValueError:
                    raise ValueError(f"invalid user ID in admin_user_ids: {item}")
        if not self.admin_user_ids:
            raise ValueError("admin_user_ids must contain at least one valid user ID (via --admin-user-ids or ADMIN_USERIDS)")

        self.state_file = parsed_args.state_file.strip()
        self.memory_file = parsed_args.memory_file.strip()

        self.model_name = parsed_args.model_name.strip()
        self.model_api_key = parsed_args.model_api_key.strip()
        if not self.model_api_key:
            raise ValueError("model_api_key must be non-empty (via --model-api-key, MODEL_API_KEY, or LLM_API_KEY)")

        self.model_api_base = parsed_args.model_api_base.strip()
        self.model_thinking_level = parsed_args.model_thinking_level.strip()
        self.model_large_name = parsed_args.model_large_name.strip()
        self.model_large_api_key = parsed_args.model_large_api_key.strip()
        self.model_large_api_base = parsed_args.model_large_api_base.strip()
        self.model_large_thinking_level = parsed_args.model_large_thinking_level.strip()
        self.model_image_name = parsed_args.model_image_name.strip()
        self.model_image_api_key = parsed_args.model_image_api_key.strip()
        self.model_image_api_base = parsed_args.model_image_api_base.strip()
        self.model_image_size = parsed_args.model_image_size.strip()
        self.model_image_thinking_level = parsed_args.model_image_thinking_level.strip()

    def is_admin(self, user_id: int) -> bool:
        return user_id in self.admin_user_ids

    def is_group_authorized(self, chat_id: int) -> bool:
        return chat_id == self.group_chat_id
