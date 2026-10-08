import pytest

import webtools
from webtools import fetch_url, html_to_text, parse_ddg_results, web_search

DDG_SAMPLE = """<!DOCTYPE html>
<html><body>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.idokep.hu%2Felorejelzes%2FBudapest&amp;rut=9f2">Időkép - <b>Budapest</b> időjárás</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.idokep.hu%2F">Részletes előrejelzés &amp; radar.</a>
</div>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="https://weather.example.com/budapest?unit=c">Weather in Budapest</a>
</div>
</body></html>"""


class _FakeResponse:
    def __init__(self, status, text):
        self.status = status
        self._text = text

    async def text(self):
        return self._text


class _FakeContent:
    """Consuming reader; `chunk_size` mimics StreamReader.read(n) returning short reads."""

    def __init__(self, body, chunk_size=None):
        self._body = body
        self._chunk_size = chunk_size
        self._pos = 0

    async def read(self, n=-1):
        if self._pos >= len(self._body):
            return b""
        size = len(self._body) - self._pos
        if n >= 0:
            size = min(size, n)
        if self._chunk_size is not None:
            size = min(size, self._chunk_size)
        chunk = self._body[self._pos:self._pos + size]
        self._pos += size
        return chunk


class _FakeFetchResponse(_FakeResponse):
    def __init__(self, status, body=b"", content_type="text/html; charset=utf-8", content=None):
        super().__init__(status, "")
        self.headers = {"Content-Type": content_type}
        self.charset = "utf-8"
        self.content = content if content is not None else _FakeContent(body)


class _FakeGet:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc_info):
        return False


class _StubSession:
    """Minimal aiohttp.ClientSession stand-in; records get() calls, no network I/O."""

    def __init__(self, response):
        self._response = response
        self.get_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return _FakeGet(self._response)


def test_parse_ddg_results():
    results = parse_ddg_results(DDG_SAMPLE)
    assert results == [
        {
            "title": "Időkép - Budapest időjárás",
            "url": "https://www.idokep.hu/elorejelzes/Budapest",
            "snippet": "Részletes előrejelzés & radar.",
        },
        {
            "title": "Weather in Budapest",
            "url": "https://weather.example.com/budapest?unit=c",
            "snippet": "",
        },
    ]
    assert parse_ddg_results(DDG_SAMPLE, max_results=1) == [results[0]]
    assert parse_ddg_results("") == []
    assert parse_ddg_results("<html><body>no result markup here</body></html>") == []


def test_html_to_text():
    html = (
        "<html><head><style>body { color: red; }</style>"
        '<script>var x = "<p>not text</p>";</script></head>'
        "<body><h1>Hello&nbsp;World</h1><p>Line one</p>\n\n<p>Line&nbsp;two &amp; more</p></body></html>"
    )
    assert html_to_text(html) == "Hello World Line one Line two & more"
    assert html_to_text("") == ""
    assert html_to_text("<p>abcdefghij</p>", max_chars=4) == "abcd"
    # Truncated document: an unterminated trailing script block must not leak JS into the text
    truncated = "<html><body><p>Sunny</p><script>var x = 1; window.bundle = {a: 1};"
    assert html_to_text(truncated) == "Sunny"


@pytest.mark.asyncio
async def test_web_search_parses_200_response():
    session = _StubSession(_FakeResponse(200, DDG_SAMPLE))
    results = await web_search(session, "weather Budapest")
    assert results[0]["url"] == "https://www.idokep.hu/elorejelzes/Budapest"
    assert results[0]["title"] == "Időkép - Budapest időjárás"
    assert session.get_calls[0][0] == webtools.WEB_SEARCH_URL
    assert session.get_calls[0][1]["params"] == {"q": "weather Budapest"}
    assert session.get_calls[0][1]["headers"]["User-Agent"] == webtools.WEB_USER_AGENT


@pytest.mark.asyncio
async def test_web_search_returns_empty_on_non_200():
    session = _StubSession(_FakeResponse(503, "unavailable"))
    assert await web_search(session, "weather Budapest") == []


@pytest.mark.asyncio
async def test_fetch_url_rejects_non_http_scheme():
    session = _StubSession(_FakeResponse(200, "irrelevant"))
    assert await fetch_url(session, "ftp://example.com/file") == ""
    assert session.get_calls == []


@pytest.mark.asyncio
async def test_fetch_url_returns_empty_for_non_text_content():
    session = _StubSession(_FakeFetchResponse(200, b"\x00\x01\x02", content_type="application/octet-stream"))
    assert await fetch_url(session, "https://example.com/data.bin") == ""


@pytest.mark.asyncio
async def test_fetch_url_extracts_readable_text():
    body = b"<html><body><script>x=1;</script><p>Sunny &amp; warm</p></body></html>"
    session = _StubSession(_FakeFetchResponse(200, body))
    assert await fetch_url(session, "https://example.com/page") == "Sunny & warm"


@pytest.mark.asyncio
async def test_fetch_url_accumulates_short_reads():
    body = b"<html><body><p>" + b"a" * 100 + b"</p></body></html>"
    response = _FakeFetchResponse(200, content=_FakeContent(body, chunk_size=8))
    session = _StubSession(response)
    assert await fetch_url(session, "https://example.com/page") == "a" * 100
