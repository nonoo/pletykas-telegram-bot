import asyncio
import base64
import io
import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from google import genai
from google.genai import types
from PIL import Image

from params import Params
from state import StateManager

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 200_000

def compress_image(image_bytes: bytes, max_dim: int = 1280, quality: int = 85) -> bytes:
    """Downscales and compresses images to max_dim and JPEG format to prevent bloat."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_dim:
                ratio = max_dim / float(max(w, h))
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=quality, optimize=True)
            return out.getvalue()
    except Exception as e:
        logger.warning("Image compression failed, using original bytes: %s", e)
        return image_bytes



class LLMClient:
    def __init__(self, params: Params, state: StateManager):
        self.params = params
        self.state = state
        self._session: Optional[aiohttp.ClientSession] = None
        self._genai_clients: Dict[str, genai.Client] = {}

    def _log_debug_payload(self, title: str, text: str) -> None:
        if self.state.is_debug_mode():
            sys.stdout.write(f"\n--- [DEBUG {title}] ---\n{text}\n--- [END DEBUG {title}] ---\n")
            sys.stdout.flush()
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_genai_client(self, api_key: str) -> genai.Client:
        key = api_key.strip()
        if key not in self._genai_clients:
            self._genai_clients[key] = genai.Client(api_key=key)
        return self._genai_clients[key]

    def _is_genai_model(self, model_name: str, api_base: str) -> bool:
        """Determines whether a model should be called via Google GenAI SDK."""
        if api_base and "googleapis.com" not in api_base.lower() and "generativelanguage" not in api_base.lower():
            return False
        name = model_name.lower()
        return name.startswith("gemini") or name.startswith("gemma") or name.startswith("imagen")

    def _get_thinking_instruction(self, thinking_level: str) -> str:
        """Returns prompt instruction according to configured thinking level."""
        tl = thinking_level.strip().lower()
        if not tl or tl in ("off", "none", "0", "disabled", "false"):
            return "Do not think or use internal reasoning. Answer directly without preamble or chain-of-thought."
        return f"Use a {thinking_level.strip()} level of internal thinking/reasoning before answering."

    def _build_thinking_config(self, thinking_level: str) -> Optional[types.ThinkingConfig]:
        """Constructs Google GenAI ThinkingConfig based on thinking level."""
        tl = thinking_level.strip().lower()
        if not tl or tl in ("off", "none", "0", "disabled", "false"):
            return types.ThinkingConfig(thinking_budget=0)
        if tl.isdigit():
            return types.ThinkingConfig(thinking_budget=int(tl))
        tl_upper = tl.upper()
        if tl_upper in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
            return types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, tl_upper))
        return None

    async def _call_openai_compatible(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        thinking_level: str = "",
    ) -> Tuple[str, int, int]:
        """Calls an OpenAI/DeepSeek compatible /chat/completions endpoint."""
        session = await self._get_session()
        url = api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        tl = thinking_level.strip().lower()
        if tl in ("low", "medium", "high"):
            payload["reasoning_effort"] = tl
        elif not tl or tl in ("off", "none", "0", "disabled", "false"):
            payload["reasoning_effort"] = "none"
        elif tl.isdigit():
            payload["thinking"] = {"type": "enabled", "budget_tokens": int(tl)}

        if self.state.is_debug_mode():
            self._log_debug_payload(f"LLM REQUEST: {url}", json.dumps(payload, indent=2, ensure_ascii=False))

        is_retry = False
        final_resp_text = ""
        final_status = 200

        async with session.post(url, headers=headers, json=payload) as resp:
            resp_text = await resp.text()
            if resp.status == 400:
                if self.state.is_debug_mode():
                    self._log_debug_payload(f"LLM RESPONSE ({resp.status}): {url}", resp_text)
                text_err = resp_text
                retry_needed = False

                # 1. If provider rejects max_tokens as out-of-bounds, cap to model maximum
                match = re.search(r"valid range of max_tokens is \[\s*\d+\s*,\s*(\d+)\s*\]", text_err)
                if not match:
                    match = re.search(r"(?:at most|less than or equal to|max(?:imum)? of)\s*(\d+)", text_err)
                if match:
                    capped = int(match.group(1))
                    logger.info("Capping max_tokens from %d to %d based on provider error", payload.get("max_tokens", 0), capped)
                    payload["max_tokens"] = capped
                    retry_needed = True

                # 2. If provider rejects thinking config parameters
                if "reasoning_effort" in payload or "extra_body" in payload or "thinking" in payload:
                    payload.pop("reasoning_effort", None)
                    payload.pop("extra_body", None)
                    payload.pop("thinking", None)
                    retry_needed = True

                if retry_needed:
                    if self.state.is_debug_mode():
                        self._log_debug_payload(f"LLM RETRY REQUEST: {url}", json.dumps(payload, indent=2, ensure_ascii=False))
                    async with session.post(url, headers=headers, json=payload) as retry_resp:
                        retry_text = await retry_resp.text()
                        if retry_resp.status != 200:
                            if self.state.is_debug_mode():
                                self._log_debug_payload(f"LLM RETRY RESPONSE ({retry_resp.status}): {url}", retry_text)
                            raise RuntimeError(f"OpenAI compatible API error {retry_resp.status}: {retry_text}")
                        data = json.loads(retry_text)
                        final_status = retry_resp.status
                        final_resp_text = retry_text
                        is_retry = True
                else:
                    raise RuntimeError(f"OpenAI compatible API error {resp.status}: {text_err}")
            elif resp.status != 200:
                if self.state.is_debug_mode():
                    self._log_debug_payload(f"LLM RESPONSE ({resp.status}): {url}", resp_text)
                raise RuntimeError(f"OpenAI compatible API error {resp.status}: {resp_text}")
            else:
                data = json.loads(resp_text)
                final_status = resp.status
                final_resp_text = resp_text

        content = ""
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "") or ""

        if self.state.is_debug_mode():
            raw_clean = final_resp_text.rstrip("\r\n")
            dbg_text = f"{raw_clean}\n{content}" if content else raw_clean
            title = f"LLM RETRY RESPONSE ({final_status}): {url}" if is_retry else f"LLM RESPONSE ({final_status}): {url}"
            self._log_debug_payload(title, dbg_text)
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0

        return content, prompt_tokens, completion_tokens

    async def _call_genai(
        self,
        api_key: str,
        model_name: str,
        contents: Any,
        system_instruction: Optional[str] = None,
        use_search_grounding: bool = False,
        thinking_level: str = "",
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        """Calls Google GenAI SDK asynchronously."""
        client = self._get_genai_client(api_key)

        tools = None
        if use_search_grounding and not model_name.lower().startswith("gemma"):
            tools = [types.Tool(google_search=types.GoogleSearch())]

        thinking_config = self._build_thinking_config(thinking_level)
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_instruction,
            "tools": tools,
            "thinking_config": thinking_config,
        }
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens
        config = types.GenerateContentConfig(**config_kwargs)
        if self.state.is_debug_mode():
            sdk_dbg = {
                "model": model_name,
                "contents": str(contents),
                "system_instruction": system_instruction,
                "thinking_level": thinking_level,
                "max_output_tokens": max_output_tokens,
            }
            self._log_debug_payload(f"GENAI SDK REQUEST ({model_name})", json.dumps(sdk_dbg, indent=2, default=str))

        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            # If search grounding failed or unsupported, disable in state and retry without tools
            if tools is not None and "search" in str(e).lower():
                logger.warning("Search grounding failed for model %s: %s. Disabling and retrying...", model_name, e)
                self.state.set_search_grounding_active(False)
                config.tools = None
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
            elif config.thinking_config is not None and ("thinking" in str(e).lower() or "budget" in str(e).lower()):
                logger.warning("Thinking config not supported for model %s: %s. Retrying without thinking_config...", model_name, e)
                config.thinking_config = None
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
            else:
                raise

        content = response.text or ""
        if self.state.is_debug_mode():
            raw_resp = ""
            try:
                raw_resp = response.model_dump_json(exclude_none=True)
            except Exception:
                raw_resp = str(response)
            raw_clean = raw_resp.rstrip("\r\n")
            dbg_text = f"{raw_clean}\n{content}" if content else raw_clean
            self._log_debug_payload(f"GENAI SDK RESPONSE ({model_name})", dbg_text)

        p_tokens = 0
        c_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p_tokens = response.usage_metadata.prompt_token_count or 0
            c_tokens = response.usage_metadata.candidates_token_count or 0

        return content, p_tokens, c_tokens

    def _extract_reaction(self, text: str) -> Tuple[str, Optional[Tuple[str, Optional[int]]]]:
        """Extracts <REACTION:emoji> or <REACTION:emoji:msg_id> and strips it from text."""
        reaction_match = re.search(r"<REACTION:\s*([^:>]+)(?::(\d+))?\s*>", text)
        reaction = None
        if reaction_match:
            emoji = reaction_match.group(1).strip()
            target_id = int(reaction_match.group(2)) if reaction_match.group(2) else None
            reaction = (emoji, target_id)
            text = re.sub(r"<REACTION:\s*[^:>]+(?::\d+)?\s*>", "", text)
        return text.strip(), reaction

    def _extract_generate_image(self, text: str) -> Tuple[str, Optional[Dict[str, str]]]:
        """Extracts <GENERATE_IMAGE>...</GENERATE_IMAGE> block and strips it from text."""
        pattern = r"<GENERATE_IMAGE>([\s\S]*?)</GENERATE_IMAGE>"
        match = re.search(pattern, text)
        spec = None
        if match:
            block = match.group(1)
            prompt = ""
            caption = ""
            source = "new"
            mode = "generate"

            for line in block.splitlines():
                line = line.strip()
                if line.lower().startswith("prompt:"):
                    prompt = line[7:].strip()
                elif line.lower().startswith("caption:"):
                    caption = line[8:].strip()
                elif line.lower().startswith("source:"):
                    source = line[7:].strip()
                elif line.lower().startswith("mode:"):
                    mode = line[5:].strip().lower()

            if prompt:
                spec = {
                    "prompt": prompt,
                    "caption": caption,
                    "source": source,
                    "mode": mode,
                }
            text = re.sub(pattern, "", text)
        return text.strip(), spec

    def _extract_image_description(self, text: str) -> Tuple[str, str]:
        """Extracts <IMAGE_DESCRIPTION>...</IMAGE_DESCRIPTION> and strips it from text."""
        pattern = r"<IMAGE_DESCRIPTION>([\s\S]*?)</IMAGE_DESCRIPTION>"
        match = re.search(pattern, text)
        desc = ""
        if match:
            desc = match.group(1).strip()
            text = re.sub(pattern, "", text)
        return text.strip(), desc
    def _check_incapable_retry(self, response_text: str) -> Tuple[bool, str]:
        """Detects if model indicated it cannot fulfill the request (e.g. needs web search / large model)."""
        if not response_text:
            return False, ""

        # 1. Explicit tag: <RETRY_WITH_LARGE_MODEL> or <NEED_LARGE_MODEL>
        m = re.search(r"<(?:RETRY_WITH_LARGE_MODEL|NEED_LARGE_MODEL)(?::\s*([^>]*))?>", response_text, re.IGNORECASE)
        if m:
            reason = m.group(1).strip() if m.group(1) else "tag requested large model"
            return True, reason

        # 2. Natural language capability refusals (e.g. web search / real-time data inability)
        if self.state.is_search_grounding_active():
            lower = response_text.lower()
            refusal_patterns = [
                r"nem tudok (?:rá)?keresni (?:az )?(?:interneten|neten|weben|google)",
                r"nincs (?:internet-?hozzáférésem|hozzáférésem (?:az )?(?:internethez|nethez|weben|google|valós idejű))",
                r"nem rendelkezem (?:internetes|valós idejű|internet-?hozzáféréssel)",
                r"i (?:cannot|can't|am unable to) (?:search|browse|access) (?:the )?(?:web|internet|google)",
                r"i do(?:n't| not) have (?:access to (?:the )?(?:internet|web|real-time)|internet access)",
                r"i do(?:n't| not) have real-?time (?:information|data|access)",
            ]
            for pat in refusal_patterns:
                if re.search(pat, lower):
                    return True, "detected inability to search web / real-time data"

        return False, ""


    def _build_evaluation_instructions(self, is_direct_trigger: bool, talkativeness: int) -> str:
        if is_direct_trigger:
            trigger_instruction = "You are directly addressed or replied to in the chat. Provide your in-character response to the group now."
        else:
            if talkativeness <= 2:
                level_str = (
                    "EXTREMELY QUIET & PASSIVE. Your default output MUST BE <NO_REPLY> in 95% of cases. "
                    "Do NOT output a text reply unless someone explicitly mentions you or asks a question directly targeted at you. "
                    "Do NOT overuse emojis—react with <REACTION:emoji> VERY RARELY (less than 5% of messages). "
                    "Never do both (text + reaction) when talkativeness is this low."
                )
            elif talkativeness <= 4:
                level_str = (
                    "Reserved. Default heavily to <NO_REPLY> in most cases (~80%). "
                    "Only chime in with text or an emoji reaction if a topic directly aligns with your core interests. "
                    "Prefer <NO_REPLY> over reacting."
                )
            elif talkativeness <= 6:
                level_str = "Balanced (Default). Participate naturally when you have something relevant, witty, or helpful to say. Otherwise output <NO_REPLY>."
            elif talkativeness <= 8:
                level_str = "Talkative. Participate actively in group discussions, frequently sharing humor, reactions, gossip, and insights."
            else:
                level_str = "Hyperactive & outspoken. Jump into almost every conversation with jokes, gossip, reactions, or banter; rarely stay silent."

            trigger_instruction = (
                f"You are passively observing the ongoing group chat. Your talkativeness setting is {talkativeness}/10.\n"
                f"Behavior instructions: {level_str}\n\n"
                "EVALUATION STEP:\n"
                "1. Ask yourself: Was I directly mentioned or asked something? If NO, your response should almost certainly be <NO_REPLY>.\n"
                "2. Choose EXACTLY ONE action: Output <NO_REPLY>, OR output a single <REACTION:emoji>, OR write a text reply. DO NOT combine text and reaction unless explicitly needed."
            )

        instructions = f"""[Instruction]
{trigger_instruction}

[Image Generation & Modification Capability]
If a user asks you to generate a new image or modify an existing image (either by referring to an image from the conversation or by replying to an image message):
Output a special image block:
<GENERATE_IMAGE>
Prompt: <detailed English visual prompt describing the desired image or modifications>
Caption: <your in-character message or response in the chat language to accompany the image>
Source: <message ID if referring to an image in transcript, or 'reply' if replying to a photo message, or 'new'>
Mode: <'modify' if changing an existing image, or 'generate' if creating a fresh image>
</GENERATE_IMAGE>

[Capability Escalation & Delegation]
If a user asks for something you are incapable of doing (such as real-time web search or Google Search for up-to-date facts, current news, live sports, weather, recent events, or information beyond your knowledge cutoff), you MUST output:
<RETRY_WITH_LARGE_MODEL>
or
<RETRY_WITH_LARGE_MODEL:reason>
Do not guess, hallucinate, or state that you cannot search the internet or lack tools. Output `<RETRY_WITH_LARGE_MODEL>` so the request is automatically delegated to a capable model with Google Search grounding.

[Output Rules]
- You can speak, react with an emoji, do BOTH, request image generation, or remain silent.
- To react with an emoji, include `<REACTION:emoji>` (e.g. `<REACTION:🔥>` or `<REACTION:🤣:1042>`). You MUST only use standard Telegram reaction emojis: 👍, 👎, ❤, 🔥, 🥰, 👏, 😁, 🤔, 🤯, 😱, 🤬, 😢, 🎉, 🤩, 🤮, 💩, 🙏, 👌, 🕊, 🤡, 🥱, 🥴, 😍, 🐳, 💯, 🤣, ⚡, 🏆, 💔, 🤨, 😐, 🍓, 🍾, 💋, 😈, 😴, 😭, 🤓, 👻, 👀, 🎃, 🙈, 😇, 😨, 🤝, 🤗, 🫡, 🤪, 🗿, 🆒, 💘, 🦄, 😘, 😎, 👾, 🤷, 😡. Note: Telegram does not support smirks (😏), winks (😉), or laughs (😂, 😄) as reactions; for cheeky/smug/flirty reactions use 😈, 😎, 💅, or 😘 instead.
- If you lack external capabilities or tools to answer (e.g. real-time web/Google search needed), output '<RETRY_WITH_LARGE_MODEL>'.
- If you do not want to intervene or say anything at all, output EXACTLY '<NO_REPLY>'.
- Never explain your decision or output meta-commentary. Speak strictly in character."""
        return instructions

    async def evaluate_and_reply(
        self,
        system_prompt: str,
        memory_context: str,
        transcript: str,
        bot_username: str,
        is_direct_trigger: bool,
        talkativeness: int = 5,
        image_bytes: Optional[bytes] = None,
    ) -> Tuple[Optional[str], Optional[Tuple[str, Optional[int]]], Optional[Dict[str, str]]]:
        """Evaluates conversation and produces in-character text, emoji reaction, and/or image generation spec."""
        current_time_str = self.state.get_current_time_str()
        timezone_str = self.state.get_timezone()
        instructions = self._build_evaluation_instructions(is_direct_trigger, talkativeness)

        user_content_prompt = f"""[Context Information]
Current Time: {current_time_str}
Timezone: {timezone_str}

[Current Memory]
{memory_context}

[Recent Conversation Transcript]
{transcript}

{instructions}"""

        model_name = self.params.model_name
        api_key = self.params.model_api_key
        api_base = self.params.model_api_base

        raw_response = ""
        prompt_tokens = 0
        completion_tokens = 0

        async def _invoke_model(
            m_name: str,
            m_key: str,
            m_base: str,
            m_tl: str,
            img_bytes: Optional[bytes] = None,
        ) -> Tuple[str, int, int]:
            if img_bytes:
                try:
                    return await self._call_vision_model(
                        model_name=m_name,
                        api_key=m_key,
                        api_base=m_base,
                        thinking_level=m_tl,
                        image_bytes=img_bytes,
                        system_prompt=system_prompt,
                        user_content_prompt=user_content_prompt,
                        use_search_grounding=self.state.is_search_grounding_active(),
                    )
                except Exception as e:
                    logger.warning("Vision call failed on model %s, falling back to text: %s", m_name, e)

            if self._is_genai_model(m_name, m_base):
                return await self._call_genai(
                    api_key=m_key,
                    model_name=m_name,
                    contents=[user_content_prompt],
                    system_instruction=f"{system_prompt}\n\n[Thinking Instruction]\n{self._get_thinking_instruction(m_tl)}",
                    use_search_grounding=self.state.is_search_grounding_active(),
                    thinking_level=m_tl,
                )
            else:
                messages = [
                    {"role": "system", "content": f"{system_prompt}\n\n[Thinking Instruction]\n{self._get_thinking_instruction(m_tl)}"},
                    {"role": "user", "content": user_content_prompt},
                ]
                return await self._call_openai_compatible(
                    api_base=m_base,
                    api_key=m_key,
                    model_name=m_name,
                    messages=messages,
                    thinking_level=m_tl,
                )
        # 1. Primary (small) model invocation
        raw_response, prompt_tokens, completion_tokens = await _invoke_model(
            model_name, api_key, api_base, self.params.model_thinking_level, image_bytes
        )

        # 2. Check if primary model indicated incapability (e.g. needs web search / large model)
        needs_retry, retry_reason = self._check_incapable_retry(raw_response)
        if needs_retry:
            large_name = self.params.model_large_name or model_name
            large_key = self.params.effective_large_api_key
            large_base = self.params.effective_large_api_base
            large_tl = self.params.effective_large_thinking_level
            logger.info(
                "Primary model (%s) indicated incapability (%s). Retrying with large model (%s)...",
                model_name,
                retry_reason,
                large_name,
            )
            retry_img_bytes = image_bytes
            if not retry_img_bytes:
                reason_lower = (retry_reason or "").lower()
                vision_keywords = ["image", "vision", "photo", "kép", "kep", "fotó", "foto"]
                if any(kw in reason_lower for kw in vision_keywords):
                    for item in reversed(self.state.get_chat_history()):
                        if item.get("media_type") == "photo" and item.get("media_b64"):
                            try:
                                retry_img_bytes = base64.b64decode(item["media_b64"])
                                logger.info("Retrieved recent photo from chat history (ID: %s) to pass to large model", item.get("id"))
                                break
                            except Exception as e:
                                logger.warning("Failed to decode photo bytes from chat history: %s", e)

            raw_response, prompt_tokens, completion_tokens = await _invoke_model(
                large_name, large_key, large_base, large_tl, retry_img_bytes
            )

        raw_trimmed = raw_response.strip()
        if raw_trimmed == "<NO_REPLY>":
            return None, None, None

        cleaned_text, image_spec = self._extract_generate_image(raw_trimmed)
        cleaned_text, reaction = self._extract_reaction(cleaned_text)
        # Clean any remaining retry tag in case large model returned one
        cleaned_text = re.sub(r"<(?:RETRY_WITH_LARGE_MODEL|NEED_LARGE_MODEL)(?::\s*[^>]*?)?>", "", cleaned_text, flags=re.IGNORECASE).strip()

        if cleaned_text == "<NO_REPLY>":
            text = None
        elif not cleaned_text:
            if is_direct_trigger:
                text = "hmm, ezen most kicsit gondolkodnom kell... 🤔"
            else:
                text = None
        else:
            text = cleaned_text

        if text is None and reaction is None and image_spec is None:
            return None, None, None

        return text, reaction, image_spec
    async def _call_vision_model(
        self,
        model_name: str,
        api_key: str,
        api_base: str,
        thinking_level: str,
        image_bytes: bytes,
        system_prompt: str,
        user_content_prompt: str,
        use_search_grounding: bool,
    ) -> Tuple[str, int, int]:
        """Dispatches an image and prompt to either Google GenAI or OpenAI-compatible vision endpoint."""
        thinking_inst = self._get_thinking_instruction(thinking_level)
        system_instruction = f"{system_prompt}\n\n[Thinking Instruction]\n{thinking_inst}"
        if self._is_genai_model(model_name, api_base):
            part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            return await self._call_genai(
                api_key=api_key,
                model_name=model_name,
                contents=[part, user_content_prompt],
                system_instruction=system_instruction,
                use_search_grounding=use_search_grounding,
                thinking_level=thinking_level,
            )
        else:
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            messages = [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                },
            ]
            return await self._call_openai_compatible(
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                messages=messages,
                thinking_level=thinking_level,
            )


    async def describe_and_reply_image(
        self,
        system_prompt: str,
        memory_context: str,
        transcript: str,
        bot_username: str,
        is_direct_trigger: bool,
        talkativeness: int,
        image_bytes: bytes,
        caption: str,
    ) -> Tuple[Optional[str], Optional[Tuple[str, Optional[int]]], Optional[Dict[str, str]], str]:
        """Dispatches photo to Large Multimodal Model to generate a reply, reaction, image spec, and visual description."""
        current_time_str = self.state.get_current_time_str()
        timezone_str = self.state.get_timezone()
        instructions = self._build_evaluation_instructions(is_direct_trigger, talkativeness)

        user_content_prompt = f"""[Context Information]
Current Time: {current_time_str}
Timezone: {timezone_str}

[Current Memory]
{memory_context}

[Recent Conversation Transcript]
{transcript}

[Photo Caption from User]
{caption if caption else "(no caption provided)"}

{instructions}

[Special Image Requirement]
In addition to your response and/or emoji reaction, you MUST include a detailed, objective visual description of this photo enclosed in:
<IMAGE_DESCRIPTION>
(detailed objective description of image subjects, setting, mood, colors, and key details)
</IMAGE_DESCRIPTION>"""

        use_large = self.state.is_image_interpretation_large_model()
        raw_response = ""

        if not use_large:
            # Setting disabled: try interpreting with the primary (small) model first
            small_name = self.params.model_name
            small_key = self.params.model_api_key
            small_base = self.params.model_api_base
            small_thinking = self.params.model_thinking_level
            # Resize image to a smaller dimension to optimize bandwidth/tokens for the small model
            small_image_bytes = compress_image(image_bytes, max_dim=800, quality=80)

            try:
                logger.info("Interpreting image using small model '%s'...", small_name)
                resp_text, _, _ = await self._call_vision_model(
                    model_name=small_name,
                    api_key=small_key,
                    api_base=small_base,
                    thinking_level=small_thinking,
                    image_bytes=small_image_bytes,
                    system_prompt=system_prompt,
                    user_content_prompt=user_content_prompt,
                    use_search_grounding=False,
                )
                trimmed = resp_text.strip()
                needs_retry, retry_reason = self._check_incapable_retry(trimmed)
                if needs_retry:
                    logger.info("Small model requested large model escalation for image interpretation: %s (%s)", trimmed[:80], retry_reason)
                elif trimmed:
                    raw_response = trimmed
                    logger.info("Small model successfully interpreted image.")
            except Exception as e:
                logger.warning("Small model image interpretation failed (%s). Falling back to large model...", e)

        # If setting is enabled (use_large is True) or small model failed/requested escalation
        if not raw_response:
            large_name = self.params.model_large_name
            large_key = self.params.effective_large_api_key
            large_base = self.params.effective_large_api_base
            large_thinking = self.params.effective_large_thinking_level
            logger.info("Interpreting image using large model '%s'...", large_name)
            resp_text, _, _ = await self._call_vision_model(
                model_name=large_name,
                api_key=large_key,
                api_base=large_base,
                thinking_level=large_thinking,
                image_bytes=image_bytes,
                system_prompt=system_prompt,
                user_content_prompt=user_content_prompt,
                use_search_grounding=self.state.is_search_grounding_active(),
            )
            raw_response = resp_text.strip()

        raw_trimmed = raw_response.strip()
        cleaned_text, image_description = self._extract_image_description(raw_trimmed)
        cleaned_text, image_spec = self._extract_generate_image(cleaned_text)
        cleaned_text, reaction = self._extract_reaction(cleaned_text)

        if cleaned_text == "<NO_REPLY>" or not cleaned_text:
            text = None
        else:
            text = cleaned_text

        return text, reaction, image_spec, image_description

    def _extract_image_from_interaction(self, interaction: Any) -> Optional[bytes]:
        """Extracts decoded image bytes from a Google Interactions API response."""
        steps = getattr(interaction, "steps", None) or (interaction.get("steps", []) if isinstance(interaction, dict) else [])
        for step in steps:
            stype = getattr(step, "type", None) or (step.get("type") if isinstance(step, dict) else None)
            if stype == "model_output":
                content = getattr(step, "content", None) or (step.get("content", []) if isinstance(step, dict) else [])
                for part in content:
                    ptype = getattr(part, "type", None) or (part.get("type") if isinstance(part, dict) else None)
                    if ptype == "image":
                        data = getattr(part, "data", None) or (part.get("data") if isinstance(part, dict) else None)
                        if data:
                            if isinstance(data, bytes):
                                return data
                            return base64.b64decode(data)
        return None

    async def generate_image(self, prompt: str, base_image_bytes: Optional[bytes] = None) -> bytes:
        """Generates a fresh image or modifies an existing image."""
        img_name = self.params.model_image_name
        img_key = self.params.effective_image_api_key
        img_base = self.params.effective_image_api_base
        img_size = self.params.model_image_size

        is_gemini_image = "gemini" in img_name.lower() and "image" in img_name.lower()
        is_google_host = "generativelanguage.googleapis.com" in img_base

        # 1. Google Interactions API (REST when API base is set to Google, or native SDK when base is empty)
        if is_gemini_image:
            model_id = img_name if img_name.startswith("models/") else f"models/{img_name}"
            if base_image_bytes:
                b64_base = base64.b64encode(base_image_bytes).decode("utf-8")
                input_payload = [
                    {"type": "text", "text": prompt},
                    {"type": "image", "data": b64_base, "mime_type": "image/jpeg"},
                ]
            else:
                input_payload = prompt

            generation_config: Dict[str, Any] = {
                "temperature": 1,
                "max_output_tokens": 65536,
                "top_p": 0.95,
                "thinking_level": self.params.effective_image_thinking_level.lower() if self.params.effective_image_thinking_level else "minimal",
                "image_config": {
                    "image_size": img_size or "1K",
                },
            }

            if not img_base or not is_google_host:
                # Call via native SDK
                client = self._get_genai_client(img_key)
                if self.state.is_debug_mode():
                    dbg_payload = {
                        "model": model_id,
                        "generation_config": generation_config,
                        "response_modalities": ["image", "text"],
                    }
                    if isinstance(input_payload, list):
                        dbg_input = []
                        for part in input_payload:
                            if isinstance(part, dict) and part.get("type") == "image":
                                dbg_input.append({"type": "image", "data": f"<base64 image {len(part.get('data', ''))} chars>", "mime_type": part.get("mime_type")})
                            else:
                                dbg_input.append(part)
                        dbg_payload["input"] = dbg_input
                    else:
                        dbg_payload["input"] = input_payload
                    self._log_debug_payload(f"IMAGE REQUEST (Google Interactions SDK: {model_id})", json.dumps(dbg_payload, indent=2, ensure_ascii=False))
                interaction = await client.aio.interactions.create(
                    model=model_id,
                    input=input_payload,
                    generation_config=generation_config,
                    response_modalities=["image", "text"],
                )
                if self.state.is_debug_mode():
                    self._log_debug_payload(f"IMAGE RESPONSE (Google Interactions SDK: {model_id})", "<image data received>")
                img_bytes = self._extract_image_from_interaction(interaction)
                if img_bytes:
                    return img_bytes
                raise RuntimeError("No image data found in Google Interactions SDK response")
            else:
                # Call via Google REST Interactions endpoint
                session = await self._get_session()
                url = img_base.rstrip("/")
                if not url.endswith("/interactions"):
                    if "/openai" in url:
                        url = url.split("/openai")[0]
                    if not url.endswith("/v1beta"):
                        url = url + "/v1beta/interactions"
                    else:
                        url = url + "/interactions"
                url = f"{url}?key={img_key}"
                payload = {
                    "model": model_id,
                    "input": input_payload,
                    "generation_config": generation_config,
                    "response_modalities": ["image", "text"],
                }
                if self.state.is_debug_mode():
                    dbg_payload = dict(payload)
                    if isinstance(payload.get("input"), list):
                        dbg_input = []
                        for part in payload["input"]:
                            if isinstance(part, dict) and part.get("type") == "image":
                                dbg_input.append({"type": "image", "data": f"<base64 image {len(part.get('data', ''))} chars>", "mime_type": part.get("mime_type")})
                            else:
                                dbg_input.append(part)
                        dbg_payload["input"] = dbg_input
                    self._log_debug_payload(f"IMAGE REQUEST: {url}", json.dumps(dbg_payload, indent=2, ensure_ascii=False))
                async with session.post(url, json=payload) as resp:
                    resp_text = await resp.text()
                    if self.state.is_debug_mode():
                        self._log_debug_payload(f"IMAGE RESPONSE ({resp.status}): {url}", "<image data received>" if resp.status == 200 else resp_text)
                        raise RuntimeError(f"Google Interactions API error {resp.status}: {resp_text}")
                    res_json = json.loads(resp_text)
                img_bytes = self._extract_image_from_interaction(res_json)
                if img_bytes:
                    return img_bytes
                raise RuntimeError("No image data found in Google Interactions REST response")

        # 2. OpenAI compatible /images/generations or /images/edits
        if not self._is_genai_model(img_name, img_base):
            session = await self._get_session()
            headers = {"Authorization": f"Bearer {img_key}"}

            if base_image_bytes:
                # Modifying existing image via /images/edits
                url = img_base.rstrip("/") + "/images/edits"
                data = aiohttp.FormData()
                data.add_field("image", base_image_bytes, filename="input.png", content_type="image/png")
                data.add_field("prompt", prompt)
                data.add_field("model", img_name)
                data.add_field("response_format", "b64_json")
                if img_size:
                    openai_size = "1024x1024" if img_size.upper() == "1K" else ("1792x1024" if img_size.upper() == "2K" else img_size)
                    data.add_field("size", openai_size)
                if self.state.is_debug_mode():
                    dbg_data = {
                        "model": img_name,
                        "prompt": prompt,
                        "image": f"<base64 image {len(base_image_bytes)} bytes>",
                        "response_format": "b64_json",
                    }
                    if img_size:
                        dbg_data["size"] = openai_size
                    self._log_debug_payload(f"IMAGE REQUEST: {url}", json.dumps(dbg_data, indent=2, ensure_ascii=False))
                async with session.post(url, headers=headers, data=data) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        if self.state.is_debug_mode():
                            self._log_debug_payload(f"IMAGE RESPONSE ({resp.status}): {url}", err_text)
                        raise RuntimeError(f"OpenAI image edit error {resp.status}: {err_text}")
                    if self.state.is_debug_mode():
                        self._log_debug_payload(f"IMAGE RESPONSE ({resp.status}): {url}", "<image data received>")
                    res = await resp.json()
                # Text to image via /images/generations
                url = img_base.rstrip("/") + "/images/generations"
                payload = {
                    "model": img_name,
                    "prompt": prompt,
                    "n": 1,
                    "response_format": "b64_json",
                }
                if img_size:
                    openai_size = "1024x1024" if img_size.upper() == "1K" else ("1792x1024" if img_size.upper() == "2K" else img_size)
                    payload["size"] = openai_size
                if self.state.is_debug_mode():
                    self._log_debug_payload(f"IMAGE REQUEST: {url}", json.dumps(payload, indent=2, ensure_ascii=False))
                async with session.post(url, headers=headers, json=payload) as resp:
                    resp_text = await resp.text()
                    if self.state.is_debug_mode():
                        self._log_debug_payload(f"IMAGE RESPONSE ({resp.status}): {url}", "<image data received>" if resp.status == 200 else resp_text)
                        raise RuntimeError(f"OpenAI image generation error {resp.status}: {resp_text}")
                    res = json.loads(resp_text)


            item = res.get("data", [])[0]
            if "b64_json" in item:
                return base64.b64decode(item["b64_json"])
            elif "url" in item:
                async with session.get(item["url"]) as get_resp:
                    return await get_resp.read()
            raise RuntimeError("No image data found in response")
        # Google GenAI Imagen SDK
        client = self._get_genai_client(img_key)

        if base_image_bytes:
            if self.state.is_debug_mode():
                self._log_debug_payload(f"IMAGE REQUEST (GenAI Imagen SDK: {img_name})", json.dumps({"model": img_name, "prompt": prompt, "edit": True}, indent=2))
            try:
                raw_ref = types.RawReferenceImage(
                    reference_id=1,
                    reference_image=types.Image(image_bytes=base_image_bytes, mime_type="image/jpeg"),
                )
                response = await client.aio.models.edit_image(
                    model=img_name,
                    prompt=prompt,
                    reference_images=[raw_ref],
                    config=types.EditImageConfig(number_of_images=1),
                )
                if self.state.is_debug_mode():
                    self._log_debug_payload(f"IMAGE RESPONSE (GenAI Imagen SDK: {img_name})", "<image data received>")
                return response.generated_images[0].image.image_bytes
            except Exception as e:
                logger.warning("edit_image failed with model %s (%s). Falling back to generate_images...", img_name, e)

        # Fresh generation
        if self.state.is_debug_mode():
            self._log_debug_payload(f"IMAGE REQUEST (GenAI Imagen SDK: {img_name})", json.dumps({"model": img_name, "prompt": prompt, "edit": False}, indent=2))
        config_kwargs: Dict[str, Any] = {
            "number_of_images": 1,
            "output_mime_type": "image/jpeg",
        }
        if img_size:
            config_kwargs["image_size"] = img_size
        try:
            response = await client.aio.models.generate_images(
                model=img_name,
                prompt=prompt,
                config=types.GenerateImagesConfig(**config_kwargs),
            )
            if self.state.is_debug_mode():
                self._log_debug_payload(f"IMAGE RESPONSE (GenAI Imagen SDK: {img_name})", "<image data received>")
            return response.generated_images[0].image.image_bytes
        except Exception as e:
            if "image_size" in config_kwargs:
                logger.warning("generate_images with image_size=%s failed (%s). Retrying without image_size...", img_size, e)
                del config_kwargs["image_size"]
                response = await client.aio.models.generate_images(
                    model=img_name,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(**config_kwargs),
                )
                if self.state.is_debug_mode():
                    self._log_debug_payload(f"IMAGE RESPONSE (GenAI Imagen SDK: {img_name})", "<image data received>")
                return response.generated_images[0].image.image_bytes
            raise
    async def generate_spontaneous_message(
        self,
        system_prompt: str,
        memory_context: str,
        transcript: str,
        timezone_str: str,
    ) -> Optional[str]:
        """Generates an unprompted spontaneous conversational message or poll."""
        current_time_str = self.state.get_current_time_str()
        prompt = f"""[System Prompt]
{system_prompt}

[Thinking Instruction]
{self._get_thinking_instruction(self.params.model_thinking_level)}

[Context Information]
Current Time: {current_time_str}
Timezone: {timezone_str}

[Current Memory]
{memory_context}

[Recent Conversation Transcript]
{transcript}

[Instruction]
You are initiating a spontaneous conversation or dropping gossip into the Telegram group chat unprompted.
You may choose between sending an engaging message or creating a group poll:
1. Regular message: Draw upon your persona, memories, group dynamics, or inside jokes. Be witty or gossipy. Output ONLY your message text.
2. Poll: If an interesting debate or group voting topic fits the context, output:
<POLL>
Question: Your poll question?
Options:
- Option 1
- Option 2
- Option 3
</POLL>
Rules: Do not refer to yourself as an AI or mention that this is automated. Speak in character."""

        model_name = self.params.model_name
        api_key = self.params.model_api_key
        api_base = self.params.model_api_base

        try:
            if self._is_genai_model(model_name, api_base):
                raw_response, p_tokens, c_tokens = await self._call_genai(
                    api_key=api_key,
                    model_name=model_name,
                    contents=[prompt],
                    system_instruction=f"{system_prompt}\n\n[Thinking Instruction]\n{self._get_thinking_instruction(self.params.model_thinking_level)}",
                    use_search_grounding=self.state.is_search_grounding_active(),
                    thinking_level=self.params.model_thinking_level,
                )
            else:
                messages = [
                    {"role": "system", "content": f"{system_prompt}\n\n[Thinking Instruction]\n{self._get_thinking_instruction(self.params.model_thinking_level)}"},
                    {"role": "user", "content": prompt},
                ]
                raw_response, p_tokens, c_tokens = await self._call_openai_compatible(
                    api_base=api_base,
                    api_key=api_key,
                    model_name=model_name,
                    messages=messages,
                    thinking_level=self.params.model_thinking_level,
                )

            res = raw_response.strip()
            return res if res and res != "<NO_REPLY>" else None
        except Exception as e:
            logger.error("Error generating spontaneous message: %s", e)
            return None

    async def curate_memory(self, current_memories: Dict[str, Any], recent_transcript: str) -> Dict[str, Any]:
        """Analyzes recent conversation history to extract long-term facts, group dynamics, and inside jokes."""
        facts_list = current_memories.get("memories", [])
        dynamics_list = current_memories.get("dynamics", [])
        jokes_list = current_memories.get("inside_jokes", [])

        current_summary = {
            "facts": [{"topic": m.get("topic"), "content": m.get("content")} for m in facts_list],
            "dynamics": [{"members": d.get("members"), "relation": d.get("relation")} for d in dynamics_list],
            "inside_jokes": [{"title": j.get("title"), "context": j.get("context")} for j in jokes_list],
        }

        prompt = f"""Review the recent conversation transcript and the current memory entries.
Your task is to update the group's long-term memory with new facts, interpersonal dynamics, and inside jokes.

[Current Memory Entries]
{json.dumps(current_summary, indent=2, ensure_ascii=False)}

[Recent Conversation Transcript]
{recent_transcript}

[Instruction]
Extract new knowledge, update outdated facts, and prune obsolete information.
Output strictly a JSON object with this exact structure:
{{
  "facts_to_add": [{{"topic": "person or subject", "content": "concise permanent fact"}}],
  "facts_to_update": [{{"topic": "existing topic", "content": "updated content"}}],
  "facts_to_discard": ["topic or content of fact to remove"],
  "dynamics_to_add": [{{"members": ["Alice", "Bob"], "relation": "relationship summary"}}],
  "dynamics_to_discard": ["members or relationship to remove"],
  "jokes_to_add": [{{"title": "joke title", "context": "lore description"}}],
  "jokes_to_discard": ["title of joke to remove"]
}}
If no changes are warranted in a category, return empty lists.

[Thinking Instruction]
{self._get_thinking_instruction(self.params.model_thinking_level)}"""

        model_name = self.params.model_name
        api_key = self.params.model_api_key
        api_base = self.params.model_api_base
        thinking_level = self.params.model_thinking_level
        raw_response = ""
        p_tokens = 0
        c_tokens = 0

        if self._is_genai_model(model_name, api_base):
            raw_response, p_tokens, c_tokens = await self._call_genai(
                api_key=api_key,
                model_name=model_name,
                contents=[prompt],
                thinking_level=thinking_level,
                max_output_tokens=200_000,
            )
        else:
            messages = [
                {"role": "system", "content": "You are a precise data curation assistant. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ]
            raw_response, p_tokens, c_tokens = await self._call_openai_compatible(
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                messages=messages,
                thinking_level=thinking_level,
                max_tokens=200_000,
            )


        default_result: Dict[str, Any] = {
            "facts_to_add": [],
            "facts_to_update": [],
            "facts_to_discard": [],
            "dynamics_to_add": [],
            "dynamics_to_discard": [],
            "jokes_to_add": [],
            "jokes_to_discard": [],
        }

        # Safe regex extraction
        json_match = re.search(r"\{[\s\S]*\}", raw_response)
        if not json_match:
            logger.warning("No JSON structure found in memory curation response: %s", raw_response)
            return default_result

        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                for k in default_result:
                    if k not in parsed or not isinstance(parsed[k], list):
                        parsed[k] = []
                return parsed
        except Exception as e:
            logger.error("Failed to parse memory curation JSON: %s (raw: %s)", e, raw_response)

        return default_result
