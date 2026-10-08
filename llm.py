import asyncio
import base64
import io
import json
import logging
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from google import genai
from google.genai import types
from PIL import Image

from params import Params
from state import StateManager
from webtools import (
    WEB_MAX_FETCHES_PER_TURN,
    WEB_MAX_SEARCHES_PER_TURN,
    fetch_url,
    web_search,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 200_000

_WEB_TOOL_TAG_RE = re.compile(r"<(?:WEB_SEARCH|FETCH_URL):\s*[^>]*>", re.IGNORECASE)

# Per-request timeouts that override the shared 120s session default.
CHAT_TIMEOUT_SEC = 10          # conversational chat/vision: fail fast on the interactive path
DEEPMEM_EMBED_TIMEOUT_SEC = 5  # deep-memory embeddings: tiny payloads, no retry
EMBED_BATCH_SIZE = 64          # provider-safe /embeddings batch chunk (Google compat layer caps at 100)

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

_QUESTION_PREFIX_RE = re.compile(
    r"^question\s*[:：\-]\s*(.*)$",
    re.IGNORECASE,
)
_OPTIONS_HEADER_RE = re.compile(
    r"^options?\s*[:：\-]?\s*$",
    re.IGNORECASE,
)
_OPTION_BULLET_RE = re.compile(
    r"^(?:[-*•–—+>]|(?:\d+|[a-zA-Z])[\.\)]|\[\d+\]|\(\d+\))\s*(.*)$"
)
_OPTION_PREFIX_RE = re.compile(
    r"^option\s*\d*\s*[:：\-]\s*(.*)$",
    re.IGNORECASE,
)


def extract_poll(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Extracts <POLL>...</POLL> block, stripping it from text and returning (cleaned_text, poll_spec)."""
    if not text:
        return text, None

    pattern = r"<POLL\b[^>]*>([\s\S]*?)(?:</POLL>|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return text.strip(), None

    block = match.group(1).strip()
    cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    question = ""
    opts: List[str] = []
    in_options = False

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _OPTIONS_HEADER_RE.match(line):
            in_options = True
            continue

        opt_match = _OPTION_BULLET_RE.match(line)
        opt_pref_match = _OPTION_PREFIX_RE.match(line)

        if opt_match:
            opt_val = opt_match.group(1).strip()
            if opt_val:
                opts.append(opt_val)
            in_options = True
            continue
        elif opt_pref_match:
            opt_val = opt_pref_match.group(1).strip()
            if opt_val:
                opts.append(opt_val)
            in_options = True
            continue

        q_match = _QUESTION_PREFIX_RE.match(line)
        if q_match:
            q_val = q_match.group(1).strip()
            if q_val:
                question = f"{question} {q_val}".strip() if question else q_val
            continue

        if in_options or len(opts) > 0:
            opts.append(line)
        else:
            question = f"{question} {line}".strip() if question else line

    question = question.strip()
    cleaned_opts: List[str] = []
    for opt in opts:
        opt_s = opt.strip().strip("\"'").strip()
        if opt_s and opt_s not in cleaned_opts:
            cleaned_opts.append(opt_s)

    # Telegram constraints: question 1-300 chars, 2-10 options, each <= 100 chars
    if not question or len(cleaned_opts) < 2:
        return cleaned_text, None

    question = question[:300]
    cleaned_opts = [o[:100] for o in cleaned_opts[:10]]

    return cleaned_text, {
        "question": question,
        "options": cleaned_opts,
    }

def extract_forget(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Extracts <FORGET>...</FORGET> or <FORGET:target> block and strips it from text."""
    if not text:
        return text, None

    inline_target = ""
    body = ""
    cleaned_text = text

    closed_pat = r"<FORGET(?::\s*([^>]+))?>([\s\S]*?)</FORGET>"
    closed_match = re.search(closed_pat, text, re.IGNORECASE)
    if closed_match:
        inline_target = (closed_match.group(1) or "").strip()
        body = (closed_match.group(2) or "").strip()
        cleaned_text = (text[:closed_match.start()] + " " + text[closed_match.end():]).strip()
    else:
        inline_pat = r"<FORGET:\s*([^>]+)>"
        inline_match = re.search(inline_pat, text, re.IGNORECASE)
        if inline_match:
            inline_target = inline_match.group(1).strip()
            cleaned_text = (text[:inline_match.start()] + " " + text[inline_match.end():]).strip()
        else:
            unclosed_pat = r"<FORGET>([\s\S]*)$"
            unclosed_match = re.search(unclosed_pat, text, re.IGNORECASE)
            if unclosed_match:
                body = (unclosed_match.group(1) or "").strip()
                cleaned_text = text[:unclosed_match.start()].strip()
            else:
                return text.strip(), None

    forget_spec: Dict[str, Any] = {
        "clear_all": False,
        "facts_to_discard": [],
        "facts_to_update": [],
        "dynamics_to_discard": [],
        "jokes_to_discard": [],
        "raw_targets": [],
    }

    if inline_target.upper() == "ALL" or body.upper() == "ALL":
        forget_spec["clear_all"] = True
        return cleaned_text, forget_spec

    if inline_target and not body:
        forget_spec["raw_targets"].append(inline_target)
        return cleaned_text, forget_spec

    current_section: Optional[str] = None
    pending_update_topic: Optional[str] = None

    for line in body.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        lower_line = line_clean.lower()
        if "facts to discard" in lower_line:
            current_section = "facts_to_discard"
            continue
        elif "facts to update" in lower_line:
            current_section = "facts_to_update"
            continue
        elif "dynamics to discard" in lower_line:
            current_section = "dynamics_to_discard"
            continue
        elif "jokes to discard" in lower_line:
            current_section = "jokes_to_discard"
            continue
        elif lower_line == "all":
            forget_spec["clear_all"] = True
            continue

        # In facts_to_update: handle Topic: ... and Content: ...
        if current_section == "facts_to_update":
            topic_match = re.match(r"^(?:-\s*)?Topic:\s*(.*)$", line_clean, re.IGNORECASE)
            content_match = re.match(r"^(?:-\s*)?Content:\s*(.*)$", line_clean, re.IGNORECASE)
            if topic_match:
                pending_update_topic = topic_match.group(1).strip()
            elif content_match and pending_update_topic:
                content_val = content_match.group(1).strip()
                forget_spec["facts_to_update"].append({
                    "topic": pending_update_topic,
                    "content": content_val,
                })
                pending_update_topic = None
            else:
                single_match = re.match(r"^(?:-\s*)?(?:Topic:\s*)?([^:]+):\s*(.*)$", line_clean, re.IGNORECASE)
                if single_match:
                    top = single_match.group(1).strip()
                    con = single_match.group(2).strip()
                    if top and con:
                        forget_spec["facts_to_update"].append({"topic": top, "content": con})
            continue

        val = re.sub(r"^[-*•]\s*", "", line_clean).strip()
        if not val:
            continue

        if current_section == "facts_to_discard":
            forget_spec["facts_to_discard"].append(val)
        elif current_section == "dynamics_to_discard":
            forget_spec["dynamics_to_discard"].append(val)
        elif current_section == "jokes_to_discard":
            forget_spec["jokes_to_discard"].append(val)
        else:
            forget_spec["raw_targets"].append(val)

    if inline_target and inline_target not in forget_spec["raw_targets"]:
        forget_spec["raw_targets"].append(inline_target)

    return cleaned_text, forget_spec


class LLMClient:
    def __init__(self, params: Params, state: StateManager):
        self.params = params
        self.state = state
        self._local = threading.local()
        self._sessions: List[aiohttp.ClientSession] = []
        self._sessions_lock = threading.Lock()

    @property
    def _genai_clients(self) -> Dict[str, genai.Client]:
        clients = getattr(self._local, "genai_clients", None)
        if clients is None:
            clients = {}
            self._local.genai_clients = clients
        return clients

    @_genai_clients.setter
    def _genai_clients(self, val: Dict[str, genai.Client]) -> None:
        self._local.genai_clients = val

    @property
    def _session(self) -> Optional[aiohttp.ClientSession]:
        return getattr(self._local, "session", None)

    @_session.setter
    def _session(self, val: Optional[aiohttp.ClientSession]) -> None:
        self._local.session = val

    def _log_debug_payload(self, title: str, text: str) -> None:
        if self.state.is_debug_mode():
            sys.stdout.write(f"\n--- [DEBUG {title}] ---\n{text}\n--- [END DEBUG {title}] ---\n")
            sys.stdout.flush()

    async def _get_session(self) -> aiohttp.ClientSession:
        current_loop = asyncio.get_running_loop()
        local_session = getattr(self._local, "session", None)
        local_loop = getattr(self._local, "loop", None)
        if local_session is None or local_session.closed or local_loop is not current_loop:
            local_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
            self._local.session = local_session
            self._local.loop = current_loop
            with self._sessions_lock:
                self._sessions.append(local_session)
        return local_session

    async def close_thread_session(self) -> None:
        """Closes the HTTP session associated with the current worker thread."""
        local_session = getattr(self._local, "session", None)
        if local_session and not local_session.closed:
            try:
                await local_session.close()
            except Exception:
                pass
        self._local.session = None
        self._local.loop = None

    async def close(self) -> None:
        with self._sessions_lock:
            for s in self._sessions:
                if not s.closed:
                    try:
                        await s.close()
                    except Exception:
                        pass
            self._sessions.clear()

    def _get_genai_client(self, api_key: str) -> genai.Client:
        key = api_key.strip()
        clients = self._genai_clients
        if key not in clients:
            clients[key] = genai.Client(api_key=key)
        return clients[key]
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
        timeout_sec: Optional[float] = None,
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

        post_kwargs: Dict[str, Any] = {"headers": headers, "json": payload}
        if timeout_sec is not None:
            # Per-request override; when unset the session's 120s default applies.
            post_kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout_sec)

        async with session.post(url, **post_kwargs) as resp:
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
                    async with session.post(url, **post_kwargs) as retry_resp:
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

    async def _call_openai_embeddings(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        texts: List[str],
        dim: int = 0,
    ) -> List[List[float]]:
        """Embeds texts via an OpenAI-compatible /embeddings endpoint. Raises RuntimeError on any failure.

        Texts are chunked (EMBED_BATCH_SIZE) because providers cap batch size, e.g. Google's
        OpenAI-compat layer rejects batches over 100. Chunk results are concatenated in input order.
        """
        if not api_base or not api_base.strip():
            raise RuntimeError("no embed api_base configured")
        vectors: List[List[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(
                await self._post_embeddings_chunk(
                    api_base, api_key, model_name, texts[start:start + EMBED_BATCH_SIZE], dim
                )
            )
        return vectors

    async def _post_embeddings_chunk(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        texts: List[str],
        dim: int = 0,
    ) -> List[List[float]]:
        """Single /embeddings request; returns vectors in input order. Raises RuntimeError on failure."""
        session = await self._get_session()
        url = api_base.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"model": model_name, "input": texts}
        if dim > 0:
            payload["dimensions"] = dim
        if self.state.is_debug_mode():
            previews = ", ".join(f"[{len(t)}ch] {t[:80]}" for t in texts)
            self._log_debug_payload(f"EMBED REQUEST: {url}", f"model={model_name} texts={len(texts)}\n{previews}")

        req_timeout = aiohttp.ClientTimeout(total=DEEPMEM_EMBED_TIMEOUT_SEC)
        async with session.post(url, headers=headers, json=payload, timeout=req_timeout) as resp:
            resp_text = await resp.text()
            if resp.status != 200:
                if self.state.is_debug_mode():
                    self._log_debug_payload(f"EMBED RESPONSE ({resp.status}): {url}", resp_text)
                raise RuntimeError(f"Embeddings API error {resp.status}: {resp_text}")
        data = json.loads(resp_text)
        rows = sorted(data.get("data", []), key=lambda row: row.get("index", 0))
        vectors = [[float(v) for v in row.get("embedding", [])] for row in rows]
        if len(vectors) != len(texts) or any(not vec for vec in vectors):
            raise RuntimeError(
                f"embedding count/dimension mismatch: got {len(vectors)} vectors for {len(texts)} texts"
            )
        if self.state.is_debug_mode():
            self._log_debug_payload(f"EMBED RESPONSE (200): {url}", f"{len(vectors)} vectors x {len(vectors[0])} dims")
        return vectors

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds texts for deep-memory retrieval. Returns [] on ANY failure (never raises)."""
        if not texts:
            return []
        # Safety default literal must match the params.py default (model_embed_name).
        model = self.params.model_embed_name or "google/gemini-embedding-2"
        key = self.params.effective_embed_api_key
        base = self.params.effective_embed_api_base
        dim = self.params.model_embed_dim
        try:
            return await self._call_openai_embeddings(base, key, model, texts, dim)
        except Exception as e:
            logger.warning("Deep-memory embedding failed (model=%s): %s", model, e)
            return []

    async def _call_genai(
        self,
        api_key: str,
        model_name: str,
        contents: Any,
        system_instruction: Optional[str] = None,
        thinking_level: str = "",
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        """Calls Google GenAI SDK asynchronously."""
        client = self._get_genai_client(api_key)

        thinking_config = self._build_thinking_config(thinking_level)
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_instruction,
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
            if config.thinking_config is not None and ("thinking" in str(e).lower() or "budget" in str(e).lower()):
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

    def _extract_schedule(self, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Extracts <SCHEDULE:oneshot>...</SCHEDULE:oneshot>, <SCHEDULE:periodic>...</SCHEDULE:periodic>,
        or <SCHEDULE:cancel:id> block and strips it from text."""
        if not text:
            return text, None

        # 1. Check for cancellation tag
        # e.g. <SCHEDULE:cancel:sched_123> or <SCHEDULE:cancel>sched_123</SCHEDULE:cancel>
        cancel_pat = r"<SCHEDULE:cancel(?::\s*([^>]+))?>(?:([\s\S]*?)</SCHEDULE:cancel>)?"
        cancel_match = re.search(cancel_pat, text, re.IGNORECASE)
        if cancel_match:
            sched_id = (cancel_match.group(1) or cancel_match.group(2) or "").strip()
            text_cleaned = re.sub(cancel_pat, "", text, flags=re.IGNORECASE).strip()
            if sched_id:
                return text_cleaned, {"action": "cancel", "schedule_id": sched_id}

        # 2. Check for creation tag
        # e.g. <SCHEDULE:(oneshot|periodic)>...</SCHEDULE:(oneshot|periodic)>
        create_pat = r"<SCHEDULE:(oneshot|periodic)>([\s\S]*?)(?:</SCHEDULE:(?:oneshot|periodic)>|</SCHEDULE>|$)"
        create_match = re.search(create_pat, text, re.IGNORECASE)
        if create_match:
            sched_type = create_match.group(1).lower()
            block = create_match.group(2)
            text_cleaned = re.sub(create_pat, "", text, flags=re.IGNORECASE).strip()

            time_val = ""
            interval_val = ""
            start_val = ""
            desc_lines: List[str] = []
            current_key = None

            for line in block.splitlines():
                line_s = line.strip()
                if not line_s:
                    continue
                if re.match(r"^Time:\s*", line_s, re.IGNORECASE):
                    time_val = re.sub(r"^Time:\s*", "", line_s, flags=re.IGNORECASE).strip()
                    current_key = "time"
                elif re.match(r"^Interval:\s*", line_s, re.IGNORECASE):
                    interval_val = re.sub(r"^Interval:\s*", "", line_s, flags=re.IGNORECASE).strip()
                    current_key = "interval"
                elif re.match(r"^Start:\s*", line_s, re.IGNORECASE):
                    start_val = re.sub(r"^Start:\s*", "", line_s, flags=re.IGNORECASE).strip()
                    current_key = "start"
                elif re.match(r"^Description:\s*", line_s, re.IGNORECASE):
                    desc_lines.append(re.sub(r"^Description:\s*", "", line_s, flags=re.IGNORECASE).strip())
                    current_key = "desc"
                elif current_key == "desc":
                    desc_lines.append(line_s)

            desc_val = " ".join([l for l in desc_lines if l]).strip()

            spec = {
                "action": "create",
                "type": sched_type,
                "time": time_val,
                "interval": interval_val,
                "start": start_val,
                "description": desc_val,
            }
            return text_cleaned, spec

        return text.strip(), None

    def _extract_poll(self, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Extracts <POLL>...</POLL> block and strips it from text."""
        return extract_poll(text)

    def extract_poll(self, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Public helper to extract <POLL>...</POLL> block and strip it from text."""
        return extract_poll(text)

    def _extract_forget(self, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Extracts <FORGET>...</FORGET> block and strips it from text."""
        return extract_forget(text)

    def extract_forget(self, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Public helper to extract <FORGET>...</FORGET> block and strip it from text."""
        return extract_forget(text)

    def _extract_web_tools(self, text: str) -> Tuple[str, List[str], List[str]]:
        """Extracts <WEB_SEARCH:query> / <FETCH_URL:url> tags and strips them from text."""
        if not text:
            return text, [], []
        queries = [q.strip() for q in re.findall(r"<WEB_SEARCH:\s*([^>]+)>", text, re.IGNORECASE)]
        urls = [u.strip() for u in re.findall(r"<FETCH_URL:\s*([^>\s]+)>", text, re.IGNORECASE)]
        cleaned = _WEB_TOOL_TAG_RE.sub("", text)
        return cleaned.strip(), [q for q in queries if q], [u for u in urls if u]

    async def _run_web_tools(self, queries: List[str], urls: List[str]) -> str:
        """Executes web searches and page fetches, returning a formatted results block."""
        session = await self._get_session()
        sections: List[str] = []
        for query in queries[:WEB_MAX_SEARCHES_PER_TURN]:
            results = await web_search(session, query)
            if not results:
                sections.append(f'No results found for "{query}".')
                continue
            lines = [f'Search results for "{query}":']
            for index, item in enumerate(results, 1):
                lines.append(f"{index}. {item.get('title', '')}")
                lines.append(f"   {item.get('url', '')}")
                lines.append(f"   {item.get('snippet', '')}")
            sections.append("\n".join(lines))
        for url in urls[:WEB_MAX_FETCHES_PER_TURN]:
            content = await fetch_url(session, url)
            sections.append(f"Content of {url}:\n{content}" if content else f"Could not fetch {url}.")
        block = "\n\n".join(sections)
        self._log_debug_payload("WEB TOOL RESULTS", block)
        return block

    def _check_incapable_retry(self, response_text: str) -> Tuple[bool, str]:
        """Detects explicit <RETRY_WITH_LARGE_MODEL> escalation requests."""
        if not response_text:
            return False, ""

        m = re.search(r"<(?:RETRY_WITH_LARGE_MODEL|NEED_LARGE_MODEL)(?::\s*([^>]*))?>", response_text, re.IGNORECASE)
        if m:
            reason = m.group(1).strip() if m.group(1) else "tag requested large model"
            return True, reason

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

        image_capability = """[Image Generation & Modification Capability]
If a user asks you to generate a new image or modify an existing image (either by referring to an image from the conversation or by replying to an image message):
Output a special image block:
<GENERATE_IMAGE>
Prompt: <detailed English visual prompt describing the desired image or modifications>
Caption: <your in-character message or response in the chat language to accompany the image>
Source: <message ID if referring to an image in transcript, or 'reply' if replying to a photo message, or 'new'>
Mode: <'modify' if changing an existing image, or 'generate' if creating a fresh image>
</GENERATE_IMAGE>
CRITICAL PROTOCOL RULE: You MUST always keep the exact English field keywords 'Prompt:', 'Caption:', 'Source:', 'Mode:' and tag names verbatim. NEVER translate these field labels into Hungarian or any other language."""
        schedule_section = """[Scheduled Replies & Reminders]
If a user asks you to remind them, check something later, or schedule a periodic message/check:
1. For one-shot replies/reminders:
Output a schedule block:
<SCHEDULE:oneshot>
Time: <ISO datetime 'YYYY-MM-DD HH:MM:SS', time 'HH:MM', or relative offset e.g. '+2h', 'in 30m', 'in 3 days', 'in 2 months', 'in 1 year'>
Description: <Comprehensive instruction detailing what to say, who requested it, and necessary context>
</SCHEDULE:oneshot>

2. For recurring / periodic replies:
Output a schedule block:
<SCHEDULE:periodic>
Interval: <recurrence interval e.g. '30m', '12h', '1d', '3 days', '2 weeks', '1 month', '3 months', '1 year'>
Start: <optional first execution time 'YYYY-MM-DD HH:MM:SS' or relative offset>
Description: <Comprehensive instruction detailing what to say or check periodically>
</SCHEDULE:periodic>

3. To cancel an active scheduled reminder or task (refer to the IDs in [Active Scheduled Reminders & Tasks]):
Output:
<SCHEDULE:cancel:schedule_id>

IMPORTANT: In the SAME turn, write your normal in-character reply to the user confirming that you scheduled or canceled the reminder! Never leave the reply empty when scheduling or canceling.
CRITICAL PROTOCOL RULE: You MUST always keep the exact English field keywords 'Time:', 'Interval:', 'Start:', 'Description:' and tag names verbatim. NEVER translate these field labels into Hungarian or any other language."""


        forget_section = """[Forgetting & Memory Clearing Capability]
If a user in the group asks you to forget something, clear or erase information, stop remembering a detail, or retracts personal facts/lore:
You MUST output a <FORGET> block specifying the items to remove or update from [Current Memory]:
<FORGET>
Facts to Discard:
- <topic or text of fact to remove completely>
Facts to Update:
- Topic: <topic>
  Content: <new content after removing the forgotten detail>
Dynamics to Discard:
- <member name or relation to remove>
Jokes to Discard:
- <title or context to remove>
</FORGET>
If the user asks to forget all memories ("forget everything", "töröld az összes emléket", "clear all memory"), output:
<FORGET>
ALL
</FORGET>

CRITICAL PROTOCOL RULES FOR FORGETTING:
1. Always keep the English field labels verbatim: 'Facts to Discard:', 'Facts to Update:', 'Dynamics to Discard:', 'Jokes to Discard:', 'Topic:', 'Content:'. Do not translate these labels into Hungarian or any other language.
2. In the SAME turn, write your normal in-character reply to the user confirming that you forgot the information. Never output ONLY the <FORGET> block without an accompanying response.
3. Be thorough: if the request implies removing related details (e.g. "forget everything about X"), discard all relevant facts, dynamics, and jokes."""
        poll_section = """[Group Poll Capability]
If a user asks you to create, start, or post a poll, or if an interesting debate or group voting topic fits the conversation:
Output an optional short conversational message introducing the poll, followed by a poll block:
<POLL>
Question: Your poll question?
Options:
- Option 1
- Option 2
- Option 3
</POLL>
You may include both your in-character text message and the <POLL> block in the same response.
CRITICAL PROTOCOL RULE: You MUST always keep the exact English field keywords 'Question:' and 'Options:' and tag names '<POLL>' and '</POLL>' verbatim. NEVER translate these keywords into Hungarian or any other language (e.g. NEVER write 'Kérdés:' or 'Opciók:'), even though the question text and options themselves are in the chat language."""

        tools_section = """[Web Search & Page Fetch Tools]
When you need current or external information (weather, news, sports, facts beyond your knowledge), use these tools instead of guessing:
- To search the web: output `<WEB_SEARCH:your search query>` (e.g. `<WEB_SEARCH:weather Budapest today>`). You may output up to 3 search tags per turn.
- To read a page: output `<FETCH_URL:https://example.com/page>` to fetch a website's readable text content (e.g. a weather page or article found via search). You may output up to 2 fetch tags per turn.
Rules: output tool tags WITHOUT any other text in that turn; the tool results will be fed back to you and you then write your normal in-character reply incorporating them. If results are missing or useless, say so in character instead of inventing facts.
CRITICAL PROTOCOL RULE: You MUST always keep the exact tag names '<WEB_SEARCH:...>' and '<FETCH_URL:...>' verbatim. NEVER translate them."""

        escalation_section = """[Capability Escalation]
If you cannot fulfill the request because of your own model limits (for example you cannot interpret an attached image), output:
<RETRY_WITH_LARGE_MODEL>
or
<RETRY_WITH_LARGE_MODEL:reason>
Do NOT use this tag for web information needs — use the <WEB_SEARCH:...> / <FETCH_URL:...> tools above instead."""
        escalation_rule = "\n- If you lack capabilities to answer (e.g. you cannot interpret an attached image), output '<RETRY_WITH_LARGE_MODEL>'."

        output_rules = f"""[Output Rules]
- You MUST select EXACTLY ONE primary action per turn: Output '<NO_REPLY>', OR output a single '<REACTION:emoji>', OR write a short text reply. DO NOT combine a text reply and an emoji reaction in the same response.
- When scheduling or canceling a reminder via `<SCHEDULE:...>`, you MUST provide an in-character text confirmation in addition to the `<SCHEDULE:...>` block.
- When clearing or updating memory via `<FORGET>`, you MUST provide an in-character text confirmation in addition to the `<FORGET>` block.
- When creating a poll via `<POLL>`, you may include an introductory in-character text message in addition to the `<POLL>` block.
- PROTOCOL KEYWORDS RULE: When using special protocol tags (<GENERATE_IMAGE>, <SCHEDULE:...>, <POLL>, <FORGET>, <WEB_SEARCH:...>, <FETCH_URL:...>), all tag names and field labels ('Prompt:', 'Caption:', 'Source:', 'Mode:', 'Time:', 'Interval:', 'Start:', 'Description:', 'Question:', 'Options:', 'Facts to Discard:', 'Facts to Update:', 'Dynamics to Discard:', 'Jokes to Discard:', 'Topic:', 'Content:') MUST strictly remain in English verbatim. NEVER translate protocol tags or field labels into the conversation language.
- STRICT REACTION RULE: Do NOT use <REACTION:emoji> as a passive default. When talkativeness is low, '<NO_REPLY>' MUST be heavily preferred over reacting in 95% of cases. Only react if a message genuinely warrants a strong reaction.
- To react with an emoji, include `<REACTION:emoji>` (e.g. `<REACTION:🔥>` or `<REACTION:🤣:1042>`). You MUST only use standard Telegram reaction emojis: 👍, 👎, ❤, 🔥, 🥰, 👏, 😁, 🤔, 🤯, 😱, 🤬, 😢, 🎉, 🤩, 🤮, 💩, 🙏, 👌, 🕊, 🤡, 🥱, 🥴, 😍, 🐳, 💯, 🤣, ⚡, 🏆, 💔, 🤨, 😐, 🍓, 🍾, 💋, 😈, 😴, 😭, 🤓, 👻, 👀, 🎃, 🙈, 😇, 😨, 🤝, 🤗, 🫡, 🤪, 🗿, 🆒, 💘, 🦄, 😘, 😎, 👾, 🤷, 😡. Note: Telegram does not support smirks (😏), winks (😉), or laughs (😂, 😄) as reactions; for cheeky/smug/flirty reactions use 😈, 😎, 💅, or 😘 instead.{escalation_rule}
- If you do not want to intervene or say anything at all, output EXACTLY '<NO_REPLY>'.
- Never explain your decision or output meta-commentary. Speak strictly in character."""

        return f"""[Instruction]
{trigger_instruction}

{image_capability}

{schedule_section}

{forget_section}

{poll_section}
{tools_section}

{escalation_section}

{output_rules}"""

    def _build_user_content_prompt(
        self,
        memory_context: str,
        transcript: str,
        instructions: str,
    ) -> str:
        current_time_str = self.state.get_current_time_str()
        timezone_str = self.state.get_timezone()
        scheduled_context = self.state.format_scheduled_replies_context()
        return f"""[Context Information]
Current Time: {current_time_str}
Timezone: {timezone_str}

[Active Scheduled Reminders & Tasks]
{scheduled_context}

[Current Memory]
{memory_context}

[Recent Conversation Transcript]
{transcript}

{instructions}"""

    def _build_image_evaluation_prompt(
        self,
        memory_context: str,
        transcript: str,
        caption: str,
        instructions: str,
    ) -> str:
        current_time_str = self.state.get_current_time_str()
        timezone_str = self.state.get_timezone()
        scheduled_context = self.state.format_scheduled_replies_context()
        return f"""[Context Information]
Current Time: {current_time_str}
Timezone: {timezone_str}

[Active Scheduled Reminders & Tasks]
{scheduled_context}

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

    async def evaluate_and_reply(
        self,
        system_prompt: str,
        memory_context: str,
        transcript: str,
        bot_username: str,
        is_direct_trigger: bool,
        talkativeness: int = 5,
        image_bytes: Optional[bytes] = None,
    ) -> Tuple[Optional[str], Optional[Tuple[str, Optional[int]]], Optional[Dict[str, str]], Optional[Dict[str, Any]]]:
        """Evaluates conversation and produces in-character text, emoji reaction, and/or image generation spec."""
        primary_instructions = self._build_evaluation_instructions(is_direct_trigger, talkativeness)
        primary_prompt = self._build_user_content_prompt(memory_context, transcript, primary_instructions)

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
            prompt_text: str,
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
                        user_content_prompt=prompt_text,
                    )
                except Exception as e:
                    logger.warning("Vision call failed on model %s, falling back to text: %s", m_name, e)

            if self._is_genai_model(m_name, m_base):
                return await self._call_genai(
                    api_key=m_key,
                    model_name=m_name,
                    contents=[prompt_text],
                    system_instruction=f"{system_prompt}\n\n[Thinking Instruction]\n{self._get_thinking_instruction(m_tl)}",
                    thinking_level=m_tl,
                )
            else:
                messages = [
                    {"role": "system", "content": f"{system_prompt}\n\n[Thinking Instruction]\n{self._get_thinking_instruction(m_tl)}"},
                    {"role": "user", "content": prompt_text},
                ]
                return await self._call_openai_compatible(
                    api_base=m_base,
                    api_key=m_key,
                    model_name=m_name,
                    messages=messages,
                    thinking_level=m_tl,
                    timeout_sec=CHAT_TIMEOUT_SEC,
                )

        # 1. Primary (small) model invocation
        raw_response, prompt_tokens, completion_tokens = await _invoke_model(
            model_name, api_key, api_base, self.params.model_thinking_level, primary_prompt, image_bytes
        )

        # Tracks the model/prompt that produced the latest response (for the web tool follow-up round)
        active_model = (model_name, api_key, api_base, self.params.model_thinking_level)
        active_prompt = primary_prompt
        active_img_bytes = image_bytes

        # 2. Check if primary model indicated incapability (e.g. cannot interpret image / needs large model)
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

            large_instructions = self._build_evaluation_instructions(is_direct_trigger, talkativeness)
            large_prompt = self._build_user_content_prompt(memory_context, transcript, large_instructions)

            raw_response, prompt_tokens, completion_tokens = await _invoke_model(
                large_name, large_key, large_base, large_tl, large_prompt, retry_img_bytes
            )
            active_model = (large_name, large_key, large_base, large_tl)
            active_prompt = large_prompt
            active_img_bytes = retry_img_bytes

        # 3. Web tool round: execute <WEB_SEARCH:...> / <FETCH_URL:...> tags if the model requested tools
        _, tool_queries, tool_urls = self._extract_web_tools(raw_response)
        if tool_queries or tool_urls:
            tool_results = await self._run_web_tools(tool_queries, tool_urls)
            followup_prompt = (
                f"{active_prompt}\n\n[Web Tool Results]\n{tool_results}\n\n"
                "[Instruction]\nThe tool results above answer your tool request. "
                "Now write your normal in-character reply to the group incorporating them. "
                "Do NOT output any further <WEB_SEARCH:...> or <FETCH_URL:...> tags in this turn."
            )
            raw_response, prompt_tokens, completion_tokens = await _invoke_model(
                *active_model, followup_prompt, active_img_bytes
            )
            raw_response = _WEB_TOOL_TAG_RE.sub("", raw_response)

        raw_trimmed = raw_response.strip()
        if raw_trimmed == "<NO_REPLY>":
            return None, None, None, None

        cleaned_text, schedule_spec = self._extract_schedule(raw_trimmed)
        cleaned_text, image_spec = self._extract_generate_image(cleaned_text)
        cleaned_text, reaction = self._extract_reaction(cleaned_text)
        # Clean any remaining retry tag in case large model returned one
        cleaned_text = re.sub(r"<(?:RETRY_WITH_LARGE_MODEL|NEED_LARGE_MODEL)(?::\s*[^>]*?)?>", "", cleaned_text, flags=re.IGNORECASE).strip()

        if cleaned_text == "<NO_REPLY>":
            text = None
        elif not cleaned_text:
            if schedule_spec:
                if schedule_spec.get("action") == "cancel":
                    text = "Rendben, töröltem az időzítőt! 👍"
                else:
                    text = "Rendben, jegyeztem az emlékeztetőt! 😉"
            elif is_direct_trigger:
                text = "hmm, ezen most kicsit gondolkodnom kell... 🤔"
            else:
                text = None
        else:
            text = cleaned_text

        if text is None and reaction is None and image_spec is None and schedule_spec is None:
            return None, None, None, None

        return text, reaction, image_spec, schedule_spec
    async def _call_vision_model(
        self,
        model_name: str,
        api_key: str,
        api_base: str,
        thinking_level: str,
        image_bytes: bytes,
        system_prompt: str,
        user_content_prompt: str,
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
                timeout_sec=CHAT_TIMEOUT_SEC,
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
    ) -> Tuple[Optional[str], Optional[Tuple[str, Optional[int]]], Optional[Dict[str, str]], str, Optional[Dict[str, Any]]]:
        """Dispatches photo to Large Multimodal Model to generate a reply, reaction, image spec, visual description, and schedule spec."""
        use_large = self.state.is_image_interpretation_large_model()
        raw_response = ""
        # Tracks the model/prompt/image that produced the latest response (for the web tool follow-up round)
        active_model: Optional[Tuple[str, str, str, str]] = None
        active_prompt = ""
        active_img_bytes = image_bytes

        if not use_large:
            # Setting disabled: try interpreting with the primary (small) model first
            small_name = self.params.model_name
            small_key = self.params.model_api_key
            small_base = self.params.model_api_base
            small_thinking = self.params.model_thinking_level
            # Resize image to a smaller dimension to optimize bandwidth/tokens for the small model
            small_image_bytes = compress_image(image_bytes, max_dim=800, quality=80)

            small_instructions = self._build_evaluation_instructions(is_direct_trigger, talkativeness)
            small_prompt = self._build_image_evaluation_prompt(memory_context, transcript, caption, small_instructions)

            try:
                logger.info("Interpreting image using small model '%s'...", small_name)
                resp_text, _, _ = await self._call_vision_model(
                    model_name=small_name,
                    api_key=small_key,
                    api_base=small_base,
                    thinking_level=small_thinking,
                    image_bytes=small_image_bytes,
                    system_prompt=system_prompt,
                    user_content_prompt=small_prompt,
                )
                trimmed = resp_text.strip()
                needs_retry, retry_reason = self._check_incapable_retry(trimmed)
                if needs_retry:
                    logger.info("Small model requested large model escalation for image interpretation: %s (%s)", trimmed[:80], retry_reason)
                elif trimmed:
                    raw_response = trimmed
                    active_model = (small_name, small_key, small_base, small_thinking)
                    active_prompt = small_prompt
                    active_img_bytes = small_image_bytes
                    logger.info("Small model successfully interpreted image.")
            except Exception as e:
                logger.warning("Small model image interpretation failed (%s). Falling back to large model...", e)

        # If setting is enabled (use_large is True) or small model failed/requested escalation
        if not raw_response:
            large_name = self.params.model_large_name
            large_key = self.params.effective_large_api_key
            large_base = self.params.effective_large_api_base
            large_thinking = self.params.effective_large_thinking_level
            large_instructions = self._build_evaluation_instructions(is_direct_trigger, talkativeness)
            large_prompt = self._build_image_evaluation_prompt(memory_context, transcript, caption, large_instructions)

            logger.info("Interpreting image using large model '%s'...", large_name)
            resp_text, _, _ = await self._call_vision_model(
                model_name=large_name,
                api_key=large_key,
                api_base=large_base,
                thinking_level=large_thinking,
                image_bytes=image_bytes,
                system_prompt=system_prompt,
                user_content_prompt=large_prompt,
            )
            raw_response = resp_text.strip()
            active_model = (large_name, large_key, large_base, large_thinking)
            active_prompt = large_prompt
            active_img_bytes = image_bytes

        # Web tool round: execute <WEB_SEARCH:...> / <FETCH_URL:...> tags if the model requested tools
        _, tool_queries, tool_urls = self._extract_web_tools(raw_response)
        if active_model and (tool_queries or tool_urls):
            tool_results = await self._run_web_tools(tool_queries, tool_urls)
            followup_prompt = (
                f"{active_prompt}\n\n[Web Tool Results]\n{tool_results}\n\n"
                "[Instruction]\nThe tool results above answer your tool request. "
                "Now write your normal in-character reply to the group incorporating them. "
                "Do NOT output any further <WEB_SEARCH:...> or <FETCH_URL:...> tags in this turn."
            )
            raw_response, _, _ = await self._call_vision_model(
                model_name=active_model[0],
                api_key=active_model[1],
                api_base=active_model[2],
                thinking_level=active_model[3],
                image_bytes=active_img_bytes,
                system_prompt=system_prompt,
                user_content_prompt=followup_prompt,
            )
            raw_response = _WEB_TOOL_TAG_RE.sub("", raw_response).strip()

        raw_trimmed = raw_response.strip()
        cleaned_text, image_description = self._extract_image_description(raw_trimmed)
        cleaned_text, schedule_spec = self._extract_schedule(cleaned_text)
        cleaned_text, image_spec = self._extract_generate_image(cleaned_text)
        cleaned_text, reaction = self._extract_reaction(cleaned_text)

        if cleaned_text == "<NO_REPLY>":
            text = None
        elif not cleaned_text:
            if schedule_spec:
                if schedule_spec.get("action") == "cancel":
                    text = "Rendben, töröltem az időzítőt! 👍"
                else:
                    text = "Rendben, jegyeztem az emlékeztetőt! 😉"
            else:
                text = None
        else:
            text = cleaned_text

        return text, reaction, image_spec, image_description, schedule_spec

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
            else:
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
        scheduled_context = self.state.format_scheduled_replies_context()
        prompt = f"""[System Prompt]
{system_prompt}

[Thinking Instruction]
{self._get_thinking_instruction(self.params.model_thinking_level)}
[Context Information]
Current Time: {current_time_str}
Timezone: {timezone_str}

[Active Scheduled Reminders & Tasks]
{scheduled_context}

[Current Memory]
{memory_context}

[Recent Conversation Transcript]
{transcript}

[Instruction]
You are initiating a spontaneous conversation or dropping gossip into the Telegram group chat unprompted.
You may choose between sending an engaging message or creating a group poll (you can also provide an introductory conversational message before the poll):
1. Regular message: Draw upon your persona, memories, group dynamics, or inside jokes. Be witty or gossipy. Output ONLY your message text.
2. Poll: If an interesting debate or group voting topic fits the context, you can output an optional in-character text message introducing the topic, followed by:
<POLL>
Question: Your poll question?
Options:
- Option 1
- Option 2
- Option 3
</POLL>
CRITICAL PROTOCOL RULE: You MUST always keep the exact English field keywords 'Question:' and 'Options:' and tag names '<POLL>' and '</POLL>' verbatim. NEVER translate these keywords into Hungarian or any other language (e.g. NEVER write 'Kérdés:' or 'Opciók:'), even though the question text and options themselves are in the chat language.
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
                    timeout_sec=CHAT_TIMEOUT_SEC,
                )

            res = raw_response.strip()
            return res if res and res != "<NO_REPLY>" else None
        except Exception as e:
            logger.error("Error generating spontaneous message: %s", e)
            return None

    async def generate_scheduled_reply(
        self,
        system_prompt: str,
        memory_context: str,
        transcript: str,
        scheduled_description: str,
        scheduled_type: str,
        timezone_str: str,
    ) -> Optional[str]:
        """Generates a scheduled reply message based on a previously set reminder or check."""
        current_time_str = self.state.get_current_time_str()
        scheduled_context = self.state.format_scheduled_replies_context()

        prompt = f"""[System Prompt]
{system_prompt}

[Thinking Instruction]
{self._get_thinking_instruction(self.params.model_thinking_level)}

[Context Information]
Current Time: {current_time_str}
Timezone: {timezone_str}

[Active Scheduled Reminders & Tasks]
{scheduled_context}

[Current Memory]
{memory_context}

[Recent Conversation Transcript]
{transcript}

[Scheduled Task Execution]
Type: {scheduled_type}
Task Description:
{scheduled_description}

[Instruction]
A previously scheduled reminder or check has triggered. Deliver your message, reminder, or conversational contribution to the Telegram group chat based on the task description above.
Speak naturally in character as Pletykas, matching the ongoing tone and language of the group.
Do not mention timers, automation, scheduled jobs, or AI mechanisms. Speak directly and naturally as a group member fulfilling what was promised or checking in."""

        model_name = self.params.model_name
        api_key = self.params.model_api_key
        api_base = self.params.model_api_base
        system_instruction = f"{system_prompt}\n\n[Thinking Instruction]\n{self._get_thinking_instruction(self.params.model_thinking_level)}"

        async def _dispatch(user_prompt: str) -> Tuple[str, int, int]:
            if self._is_genai_model(model_name, api_base):
                return await self._call_genai(
                    api_key=api_key,
                    model_name=model_name,
                    contents=[user_prompt],
                    system_instruction=system_instruction,
                    thinking_level=self.params.model_thinking_level,
                )
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ]
            return await self._call_openai_compatible(
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                messages=messages,
                thinking_level=self.params.model_thinking_level,
                timeout_sec=CHAT_TIMEOUT_SEC,
            )

        try:
            raw_response, p_tokens, c_tokens = await _dispatch(prompt)

            # Web tool round: execute <WEB_SEARCH:...> / <FETCH_URL:...> tags if the model requested tools
            _, tool_queries, tool_urls = self._extract_web_tools(raw_response)
            if tool_queries or tool_urls:
                tool_results = await self._run_web_tools(tool_queries, tool_urls)
                followup_prompt = (
                    f"{prompt}\n\n[Web Tool Results]\n{tool_results}\n\n"
                    "[Instruction]\nThe tool results above answer your tool request. "
                    "Now write your normal in-character reply to the group incorporating them. "
                    "Do NOT output any further <WEB_SEARCH:...> or <FETCH_URL:...> tags in this turn."
                )
                raw_response, p_tokens, c_tokens = await _dispatch(followup_prompt)

            res = raw_response.strip()
            res = re.sub(r"<(?:NO_REPLY|RETRY_WITH_LARGE_MODEL|NEED_LARGE_MODEL)(?::\s*[^>]*?)?>", "", res, flags=re.IGNORECASE).strip()
            res = _WEB_TOOL_TAG_RE.sub("", res).strip()
            return res if res else None
        except Exception as e:
            logger.error("Error generating scheduled reply: %s", e)
            return None

    async def curate_memory(
        self,
        current_memories: Dict[str, Any],
        recent_transcript: str,
        deep_memories: Optional[Dict[str, Any]] = None,
        archive_candidates: Optional[Dict[str, Any]] = None,
        archive_age_days: int = 7,
    ) -> Dict[str, Any]:
        """Analyzes recent conversation history to extract long-term facts, group dynamics, and inside jokes.

        `deep_memories` is the deep (archived) store's get_all_memories() dict, shown
        as a read-only reference (None = backward-compatible, section omitted).
        `archive_candidates` is a precomputed {"facts": [...], "dynamics": [...],
        "jokes": [...]} summary of hot entries older than `archive_age_days`; the
        CALLER does the age filtering, this method only renders it, and None/empty
        omits the section. `archive_age_days` <= 0 disables the age-based rule
        (0 disables age candidacy; the archive lists themselves stay available).
        """
        facts_list = current_memories.get("memories", [])
        dynamics_list = current_memories.get("dynamics", [])
        jokes_list = current_memories.get("inside_jokes", [])

        current_summary = {
            "facts": [{"topic": m.get("topic"), "content": m.get("content")} for m in facts_list],
            "dynamics": [{"members": d.get("members"), "relation": d.get("relation")} for d in dynamics_list],
            "inside_jokes": [{"title": j.get("title"), "context": j.get("context")} for j in jokes_list],
        }

        deep_section = ""
        if deep_memories is not None:
            deep_summary = {
                "facts": [{"topic": m.get("topic"), "content": m.get("content")} for m in deep_memories.get("memories", [])],
                "dynamics": [{"members": d.get("members"), "relation": d.get("relation")} for d in deep_memories.get("dynamics", [])],
                "inside_jokes": [{"title": j.get("title"), "context": j.get("context")} for j in deep_memories.get("inside_jokes", [])],
            }
            deep_section = f"\n[Deep Memory Entries (archived, read-only reference)]\n{json.dumps(deep_summary, indent=2, ensure_ascii=False)}\n"

        candidates_section = ""
        if archive_candidates:
            candidates_section = f"\n[Archive Candidates (hot entries older than {archive_age_days} days)]\n{json.dumps(archive_candidates, indent=2, ensure_ascii=False)}\n"

        age_rule = ""
        if archive_age_days > 0:
            age_rule = (
                f"Entries older than {archive_age_days} days SHOULD be moved to the archive "
                f"(facts_to_archive / dynamics_to_archive / jokes_to_archive) unless they are still "
                f"actively referenced in the recent conversation.\n"
            )

        prompt = f"""Review the recent conversation transcript and the current memory entries.
Your task is to update the group's long-term memory with new facts, interpersonal dynamics, and inside jokes.

[Current Memory Entries]
{json.dumps(current_summary, indent=2, ensure_ascii=False)}
{deep_section}{candidates_section}
[Recent Conversation Transcript]
{recent_transcript}

[Instruction]
Extract new knowledge, update outdated facts, and prune obsolete information.
{age_rule}You may also archive entries for other reasons (obsolete or superseded information). When archived information resurfaces and is relevant again, restore it via the promote lists (facts_to_promote / dynamics_to_promote / jokes_to_promote).
NEVER re-add a fact that is already present in deep memory: return it in the corresponding promote list instead.
CRITICAL FORGETTING RULE: If anyone in the conversation asked to forget, retract, delete, or stop remembering any information, you MUST ensure that information is thoroughly discarded (placed in 'facts_to_discard', 'dynamics_to_discard', or 'jokes_to_discard', or updated via 'facts_to_update'). NEVER re-add or retain any information that a user requested to be forgotten or erased!
Output strictly a JSON object with this exact structure:
{{
  "facts_to_add": [{{"topic": "person or subject", "content": "concise permanent fact"}}],
  "facts_to_update": [{{"topic": "existing topic", "content": "updated content"}}],
  "facts_to_discard": ["topic or content of fact to remove"],
  "facts_to_archive": ["topic of active fact to move to the archive"],
  "facts_to_promote": ["topic of archived fact to restore to active memory"],
  "dynamics_to_add": [{{"members": ["Alice", "Bob"], "relation": "relationship summary"}}],
  "dynamics_to_discard": ["members or relationship to remove"],
  "dynamics_to_archive": ["members of dynamic to archive"],
  "dynamics_to_promote": ["members of archived dynamic to restore"],
  "jokes_to_add": [{{"title": "joke title", "context": "lore description"}}],
  "jokes_to_discard": ["title of joke to remove"],
  "jokes_to_archive": ["title of joke to archive"],
  "jokes_to_promote": ["title of archived joke to restore"]
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
            # No timeout: large JSON output needs headroom
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
            "facts_to_archive": [],
            "facts_to_promote": [],
            "dynamics_to_add": [],
            "dynamics_to_discard": [],
            "dynamics_to_archive": [],
            "dynamics_to_promote": [],
            "jokes_to_add": [],
            "jokes_to_discard": [],
            "jokes_to_archive": [],
            "jokes_to_promote": [],
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

    async def curate_forget(
        self,
        current_memories: Dict[str, Any],
        forget_request_text: str,
        sender_name: str = "",
        recent_transcript: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Analyzes a forget request and determines which facts, dynamics, and jokes should be discarded or updated.

        Returns None when the analysis call or its parsing fails; a parsed spec
        with empty lists is a valid "nothing to forget" result.
        """
        facts_list = current_memories.get("memories", [])
        dynamics_list = current_memories.get("dynamics", [])
        jokes_list = current_memories.get("inside_jokes", [])

        current_summary = {
            "facts": [{"topic": m.get("topic"), "content": m.get("content")} for m in facts_list],
            "dynamics": [{"members": d.get("members"), "relation": d.get("relation")} for d in dynamics_list],
            "inside_jokes": [{"title": j.get("title"), "context": j.get("context")} for j in jokes_list],
        }

        prompt = f"""A user in the group chat requested the bot to forget something.
Your task is to analyze the user's forget request against the current memory entries and determine exactly what must be cleared or updated so that the information is thoroughly removed from memory.

[Current Memory Entries]
{json.dumps(current_summary, indent=2, ensure_ascii=False)}

[Forget Request Message]
From: {sender_name or 'User'}
Message: {forget_request_text}

[Recent Context]
{recent_transcript}

Guidelines:
1. If the user asks to forget all memories ("forget everything", "töröld az összes emléket", "clear all memory"), set "clear_all": true.
2. If an entire fact should be removed, place its topic (or exact content) in "facts_to_discard".
3. If an existing fact contains multiple pieces of information and only one piece was asked to be forgotten, keep the remaining details and put the updated fact in "facts_to_update".
4. If interpersonal dynamics or inside jokes relate to the forgotten information, place them in "dynamics_to_discard" or "jokes_to_discard".
5. Be thorough: resolve personal references (e.g. "my cat" said by "Béla" refers to facts about Béla's cat).
6. If the request does not match anything in current memory, return empty lists.

Output strictly a JSON object with this exact structure:
{{
  "clear_all": false,
  "facts_to_discard": ["topic or fact to remove"],
  "facts_to_update": [{{"topic": "topic name", "content": "remaining content without forgotten detail"}}],
  "dynamics_to_discard": ["member name or relation to remove"],
  "jokes_to_discard": ["title of joke to remove"]
}}

[Thinking Instruction]
{self._get_thinking_instruction(self.params.model_thinking_level)}"""

        model_name = self.params.model_name
        api_key = self.params.model_api_key
        api_base = self.params.model_api_base
        thinking_level = self.params.model_thinking_level
        raw_response = ""

        try:
            if self._is_genai_model(model_name, api_base):
                raw_response, _, _ = await self._call_genai(
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
                # No timeout: large JSON output needs headroom
                raw_response, _, _ = await self._call_openai_compatible(
                    api_base=api_base,
                    api_key=api_key,
                    model_name=model_name,
                    messages=messages,
                    thinking_level=thinking_level,
                    max_tokens=200_000,
                )
        except Exception as e:
            logger.error("Failed model call in curate_forget: %s", e)
            return None

        json_match = re.search(r"\{[\s\S]*\}", raw_response)
        if not json_match:
            logger.warning("No JSON structure found in curate_forget response: %s", raw_response)
            return None

        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                return {
                    "clear_all": bool(parsed.get("clear_all", False)),
                    "facts_to_discard": list(parsed.get("facts_to_discard", [])) if isinstance(parsed.get("facts_to_discard"), list) else [],
                    "facts_to_update": list(parsed.get("facts_to_update", [])) if isinstance(parsed.get("facts_to_update"), list) else [],
                    "dynamics_to_discard": list(parsed.get("dynamics_to_discard", [])) if isinstance(parsed.get("dynamics_to_discard"), list) else [],
                    "jokes_to_discard": list(parsed.get("jokes_to_discard", [])) if isinstance(parsed.get("jokes_to_discard"), list) else [],
                    "raw_targets": [],
                }
        except Exception as e:
            logger.error("Failed to parse curate_forget JSON: %s (raw: %s)", e, raw_response)

        return None
