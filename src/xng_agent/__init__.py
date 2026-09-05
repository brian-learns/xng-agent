"""xng-agent: a NOOA agent that researches topics with live web search (xng / SearXNG)
and real page browsing (browser-use primitives, no sub-agent).

No memory: every run is self-contained. The model-facing surface is one class:

- ``search_web``  — deterministic, wraps the ``xng`` Python API (SearXNG).
- ``browse``      — deterministic, drives a headless browser directly with
                    browser-use session primitives (goto + text extraction);
                    no LLM involved, so the only agent loop is the NOOA one.
- ``research``    — agentic (CodeAct): the model writes Python that searches,
                    browses, and returns a validated ``ResearchReport``.

Run:
    uv run xng-agent "topic to research"
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass

import xng
from browser_use import Browser
from browser_use.actor.page import Page as BrowserPage
from nooa import Agent, strategy
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm.registry import get_llm_client
from pydantic import BaseModel, Field

# --- local LLM (llama-server router, OpenAI-compatible) ---------------------
LLM_MODEL = os.environ.get("NOOA_MODEL", "Qwen3.6-35B-A3B-MXFP4_MOE")
LLM_BASE = os.environ.get("NOOA_LLM_BASE", "http://127.0.0.1:8080/v1")

# Thinking disabled for speed (litellm passes it through to llama-server).
nooa_llm = get_llm_client(
    f"openai/{LLM_MODEL}",
    api_base=LLM_BASE,
    api_key="local",
    max_tokens=8192,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

logger = logging.getLogger(__name__)

CHROME_PATH = os.environ.get("CHROME_PATH", "/usr/bin/google-chrome")
MAX_CONTENT_CHARS = 3000
BROWSE_TIMEOUT = 30  # seconds per page operation; hung CDP calls must not stall the run
BROWSER_START_TIMEOUT = 30  # seconds to wait for the CDP connection after start()


@dataclass
class Hit:
    """One live web search result."""

    title: str
    url: str
    snippet: str
    published_date: str | None = None


@dataclass
class PageNote:
    """A page opened in the headless browser and extracted as text."""

    url: str
    title: str
    content: str


class Source(BaseModel):
    url: str = Field(description="URL of the page consulted.")
    title: str | None = Field(default=None, description="Page title, if known.")
    via_browser: bool = Field(description="True if the page was opened in the browser.")


class ResearchReport(BaseModel):
    topic: str = Field(description="The research topic.")
    summary: str = Field(description="3-5 sentence synthesis of what was found.")
    key_facts: list[str] = Field(description="Most important facts; each fact ends with its source URL in parentheses.")
    sources: list[Source] = Field(description="Pages consulted, most important first.")


class XngBrowserAgent(Agent, llm=nooa_llm):
    """You are a web research agent. Search the live web with self.search_web(),
    then open the most promising pages with self.browse() to read their real
    content, and synthesize a cited report. Append every browsed page to
    self.findings."""

    findings: list[PageNote]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.findings: list[PageNote] = []
        # One shared headless session for the whole run (private: hidden from
        # the model's doc(self) and state block).
        self._browser: Browser | None = None
        # Serializes browse(): the shared page must not be navigated
        # concurrently, and _get_browser() must not create two sessions.
        self._browser_lock = asyncio.Lock()

    # --- Deterministic tools: ordinary Python, callable by the model ---

    async def search_web(self, query: str, limit: int = 5) -> list[Hit]:
        """Search the live web via SearXNG. Returns title, url, snippet and
        published date for each hit."""
        # to_thread: xng.search is blocking HTTP and must not stall the event
        # loop that the browser awaits run on.
        resp = await asyncio.to_thread(xng.search, query, limit=limit)
        return [Hit(h.title, h.url, h.content or "", h.published_date) for h in resp.results]

    async def _kill_browser(self) -> None:
        """Kill the shared session and its Chrome process.

        browser-use's stop() deliberately keeps the browser process alive for
        reconnecting; kill() is what actually tears it down (no orphans)."""
        if self._browser is not None:
            try:
                await self._browser.kill()
            except Exception:  # session may already be dead
                logger.debug("browser kill failed (session already dead?)", exc_info=True)
            self._browser = None

    async def _get_browser(self) -> Browser:
        """Return a live shared session (lazily started).

        browser-use's start() returns before the launch handler finishes, so
        poll is_cdp_connected() (a sync WebSocket-state check) until the CDP
        connection is actually open. A session that never connects is killed
        and replaced once; we never run two sessions in parallel."""
        for attempt in (1, 2):
            if self._browser is None:
                self._browser = Browser(executable_path=CHROME_PATH, headless=True)
                try:
                    await asyncio.wait_for(self._browser.start(), timeout=BROWSER_START_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning("browser start() timed out; checking CDP anyway")
                except Exception:
                    # A half-started session is left in self._browser; the CDP
                    # poll below fails, it gets killed, and the retry replaces it.
                    logger.warning("browser start() failed; checking CDP anyway", exc_info=True)
            deadline = time.monotonic() + BROWSER_START_TIMEOUT
            while time.monotonic() < deadline:
                # is_cdp_connected is a property (bool), not a method.
                if self._browser.is_cdp_connected:
                    return self._browser
                await asyncio.sleep(0.25)
            logger.warning("browser CDP connection never came up (attempt %d)", attempt)
            await self._kill_browser()
        raise RuntimeError("headless browser did not connect within timeout")

    @staticmethod
    async def _wait_page_ready(page: BrowserPage, timeout: float) -> None:
        """Wait for the document to finish loading; CDP Page.navigate returns
        as soon as navigation starts, so the DOM is not ready when it resolves."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = await page.evaluate("() => document.readyState")
            except Exception:
                # The JS context may be torn down mid-navigation (redirects);
                # a failed probe means "still loading", not a dead session.
                state = None
            if state == "complete":
                return
            await asyncio.sleep(0.2)
        raise TimeoutError(f"page did not finish loading within {timeout}s")

    async def browse(self, url: str) -> PageNote:
        """Open *url* in the shared headless browser and return the page's
        visible text (truncated) with its title. No interaction: use this to
        read a page, not to fill forms or click through. Note the url in the
        result is the FINAL url after any redirects."""
        # Model code may fire browse() calls concurrently (e.g. via
        # asyncio.gather); serialize so one page is never navigated twice at
        # once and _get_browser() never creates a second session.
        async with self._browser_lock:
            browser = await self._get_browser()
            try:
                page = await asyncio.wait_for(browser.must_get_current_page(), timeout=BROWSE_TIMEOUT)
                await asyncio.wait_for(page.goto(url), timeout=BROWSE_TIMEOUT)
                await self._wait_page_ready(page, BROWSE_TIMEOUT)
                # Read everything from the DOM (the CDP target title can lag behind nav).
                title = (await asyncio.wait_for(page.evaluate("() => document.title"), timeout=10)).strip()
                content = (
                    await asyncio.wait_for(
                        page.evaluate("() => (document.body ? document.body.innerText : '')"), timeout=10
                    )
                ).strip()
                final_url = await asyncio.wait_for(page.get_url(), timeout=10)
            except Exception:
                # A hung or crashed session would poison every later call: kill it
                # so the next browse() starts fresh, and let the model see the error.
                await self._kill_browser()
                raise
            note = PageNote(url=final_url, title=title, content=content[:MAX_CONTENT_CHARS])
            self.findings.append(note)
            return note

    # --- Agentic method: ellipsis body, run by the LLM (CodeAct loop) ---

    # ty: the ellipsis body is intentional — NOOA implements it at runtime via the LLM.
    @strategy(CodeActStrategy())
    async def research(self, topic: str) -> ResearchReport:  # ty: ignore[empty-body]
        """Research {topic}. Search the web with await self.search_web(), pick the 2-3
        most promising URLs, and read each with await self.browse(url). Base the
        report only on the browsed content and search snippets — do not invent
        content. Cite a URL for each key fact."""
        ...


async def _run(topic: str) -> tuple[ResearchReport, int]:
    """Research *topic*, always killing the shared browser session afterwards."""
    agent = XngBrowserAgent()
    try:
        return await agent.research(topic), len(agent.findings)
    finally:
        await agent._kill_browser()


def main() -> None:
    """Run the agent on a topic from argv and print the validated report."""
    topic = " ".join(sys.argv[1:]) or "What is SearXNG and why do people self-host it?"
    report, browsed = asyncio.run(_run(topic))
    print("\nREPORT:")
    print(report.model_dump_json(indent=2))
    print(f"\nbrowsed {browsed} pages")


if __name__ == "__main__":
    main()
