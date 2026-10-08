"""Provider-independent web tools: DuckDuckGo HTML search and page text extraction.

Pure parsing helpers plus thin aiohttp wrappers used by LLMClient to execute
<WEB_SEARCH:query> / <FETCH_URL:url> protocol tags. All network functions
degrade to empty results on any failure (never raise), mirroring the
LLMClient.embed_texts philosophy.
"""

import logging
import re
import urllib.parse
from html import unescape
from typing import Dict, List

import aiohttp

logger = logging.getLogger(__name__)

WEB_SEARCH_URL = "https://html.duckduckgo.com/html/"
WEB_TIMEOUT_SEC = 10
WEB_SEARCH_RESULTS = 5
WEB_FETCH_MAX_CHARS = 4000
WEB_FETCH_MAX_BYTES = 262144  # download cap before text extraction
WEB_MAX_SEARCHES_PER_TURN = 3
WEB_MAX_FETCHES_PER_TURN = 2
WEB_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

_RESULT_LINK_RE = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
_RESULT_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
# The `$` fallback drops an unterminated trailing block left behind by the byte-cap truncation.
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?(?:</script\s*>|$)", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?(?:</style\s*>|$)", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_fragment(fragment: str) -> str:
    """Strips inner tags, unescapes entities, and collapses whitespace."""
    return _WHITESPACE_RE.sub(" ", unescape(_TAG_RE.sub("", fragment))).strip()


def _unwrap_href(href: str) -> str:
    """Resolves a DuckDuckGo result href, unwrapping the uddg redirect parameter."""
    href = unescape(href).strip()
    if not href:
        return ""
    parsed = urllib.parse.urlsplit(href)
    target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
    if target:
        return target
    if href.startswith("//"):
        return f"https:{href}"
    return href


def parse_ddg_results(html: str, max_results: int = WEB_SEARCH_RESULTS) -> List[Dict[str, str]]:
    """Parses a DuckDuckGo HTML results page into [{title, url, snippet}]. Never raises."""
    if not html:
        return []
    try:
        links = _RESULT_LINK_RE.findall(html)
        snippets = _RESULT_SNIPPET_RE.findall(html)
        results: List[Dict[str, str]] = []
        for index, (href, title) in enumerate(links[:max_results]):
            results.append({
                "title": _clean_fragment(title),
                "url": _unwrap_href(href),
                "snippet": _clean_fragment(snippets[index]) if index < len(snippets) else "",
            })
        return results
    except Exception as e:
        logger.warning("Failed to parse DuckDuckGo results: %s", e)
        return []


def html_to_text(html: str, max_chars: int = WEB_FETCH_MAX_CHARS) -> str:
    """Extracts readable text from raw HTML. Never raises."""
    if not html:
        return ""
    try:
        text = _SCRIPT_RE.sub(" ", html)
        text = _STYLE_RE.sub(" ", text)
        text = _TAG_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", unescape(text)).strip()
        return text[:max_chars]
    except Exception as e:
        logger.warning("Failed to convert HTML to text: %s", e)
        return ""


async def web_search(
    session: aiohttp.ClientSession,
    query: str,
    max_results: int = WEB_SEARCH_RESULTS,
) -> List[Dict[str, str]]:
    """Searches DuckDuckGo's HTML endpoint. Returns [] on any failure (never raises)."""
    try:
        timeout = aiohttp.ClientTimeout(total=WEB_TIMEOUT_SEC)
        async with session.get(
            WEB_SEARCH_URL,
            params={"q": query},
            headers={"User-Agent": WEB_USER_AGENT},
            timeout=timeout,
        ) as resp:
            if resp.status != 200:
                logger.warning("DuckDuckGo search returned HTTP %s for %r", resp.status, query)
                return []
            body = await resp.text()
    except Exception as e:
        logger.warning("DuckDuckGo search failed for %r: %s", query, e)
        return []
    return parse_ddg_results(body, max_results)


async def fetch_url(
    session: aiohttp.ClientSession,
    url: str,
    max_chars: int = WEB_FETCH_MAX_CHARS,
) -> str:
    """Fetches an http(s) page and returns its readable text. Returns "" on any failure (never raises)."""
    if urllib.parse.urlsplit(url).scheme.lower() not in ("http", "https"):
        logger.warning("Refusing to fetch non-HTTP URL: %r", url)
        return ""
    try:
        timeout = aiohttp.ClientTimeout(total=WEB_TIMEOUT_SEC)
        async with session.get(url, headers={"User-Agent": WEB_USER_AGENT}, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning("Fetch of %r returned HTTP %s", url, resp.status)
                return ""
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in content_type and "text/" not in content_type:
                logger.warning("Fetch of %r returned non-text content type %r", url, content_type)
                return ""
            charset = resp.charset or "utf-8"
            # StreamReader.read(n) may return short reads, so accumulate until the cap or EOF.
            chunks: List[bytes] = []
            total = 0
            while total < WEB_FETCH_MAX_BYTES:
                chunk = await resp.content.read(WEB_FETCH_MAX_BYTES - total)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            raw = b"".join(chunks)
    except Exception as e:
        logger.warning("Failed to fetch %r: %s", url, e)
        return ""
    return html_to_text(raw.decode(charset, errors="replace"), max_chars)
