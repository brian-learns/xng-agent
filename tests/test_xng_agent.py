"""Unit tests for xng_agent: no browser, no network, no LLM (all I/O is faked)."""

import asyncio
import types
from datetime import datetime

import pytest
import xng

import xng_agent
from xng_agent import XngBrowserAgent


class FakePage:
    """Mimics browser_use.actor.page.Page just enough for browse()."""

    def __init__(self, ready_states, final_url="http://example.com/final", fail_first_probe=False):
        self._ready_states = list(ready_states)
        self.final_url = final_url
        self._fail_first_probe = fail_first_probe

    async def goto(self, url):
        pass

    async def evaluate(self, expression):
        if "readyState" in expression:
            if self._fail_first_probe:
                self._fail_first_probe = False
                raise RuntimeError("execution context was destroyed")
            if self._ready_states:
                return self._ready_states.pop(0)
            return "complete"
        if "document.title" in expression:
            return "  A Title  "
        return "body text " * 2000  # longer than MAX_CONTENT_CHARS

    async def get_url(self):
        return self.final_url


class FakeBrowser:
    def __init__(self, page):
        self._page = page
        self.killed = False

    async def must_get_current_page(self):
        return self._page

    async def kill(self):
        self.killed = True


def make_agent(monkeypatch, page) -> tuple[XngBrowserAgent, FakeBrowser]:
    """An agent whose _get_browser() returns a pre-seeded FakeBrowser."""
    agent = XngBrowserAgent()
    browser = FakeBrowser(page)

    async def fake_get_browser(self):
        return browser

    monkeypatch.setattr(XngBrowserAgent, "_get_browser", fake_get_browser)
    return agent, browser


def test_browse_truncates_and_records_final_url(monkeypatch):
    agent, browser = make_agent(monkeypatch, FakePage(["loading", "complete"]))
    note = asyncio.run(agent.browse("http://example.com/start"))
    assert note.url == "http://example.com/final"
    assert note.title == "A Title"
    assert len(note.content) == xng_agent.MAX_CONTENT_CHARS
    assert agent.findings == [note]
    assert browser.killed is False


def test_browse_survives_evaluate_error_during_navigation(monkeypatch):
    """A probe that fails once (JS context torn down by a redirect) must not
    kill the shared browser session."""
    agent, browser = make_agent(monkeypatch, FakePage(["complete"], fail_first_probe=True))
    note = asyncio.run(agent.browse("http://example.com"))
    assert browser.killed is False
    assert note.title == "A Title"


def test_wait_page_ready_times_out():
    page = FakePage(["loading"] * 10_000)
    with pytest.raises(TimeoutError):
        # ty: ignore[invalid-argument-type]
        asyncio.run(XngBrowserAgent._wait_page_ready(page, 0.3))


def test_today_holds_the_current_date():
    agent = XngBrowserAgent()
    now = datetime.now().astimezone()
    assert now.strftime("%Y-%m-%d") in agent.today
    # %z renders empty on naive datetimes; the offset must actually be there.
    assert agent.today.endswith(now.strftime("%z"))


def test_search_web_maps_hits(monkeypatch):
    results = [
        types.SimpleNamespace(title="T1", url="http://a", content="snippet", published_date="2026-01-01"),
        types.SimpleNamespace(title="T2", url="http://b", content=None, published_date=None),
        types.SimpleNamespace(title="T3", url="http://c", content="x", published_date=None),
    ]

    # The real xng.search honors limit; the agent no longer re-slices.
    def fake_search(query, limit=None):
        return types.SimpleNamespace(results=results[:limit])

    monkeypatch.setattr(xng, "search", fake_search)
    hits = asyncio.run(XngBrowserAgent().search_web("query", limit=2))
    assert [h.url for h in hits] == ["http://a", "http://b"]
    assert hits[0].title == "T1"
    assert hits[0].snippet == "snippet"
    assert hits[0].published_date == "2026-01-01"
    assert hits[1].snippet == ""
