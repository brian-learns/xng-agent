"""xng-agent: a NOOA agent that researches topics with live web search (xng / SearXNG)
and real page browsing (browser-use primitives, no sub-agent).

No memory: every run is self-contained. The model-facing surface is one class
plus one skill:

- ``search_web``  — deterministic, wraps the ``xng`` Python API (SearXNG).
- ``browse``      — deterministic, drives a headless browser directly with
                    browser-use session primitives (goto + text extraction);
                    no LLM involved, so the only agent loop is the NOOA one.
  Both live in :class:`xng_agent.skills.WebResearchSkill`, a standalone tool
  belt (no Agent inheritance) registered under the ``nooa.skills`` entry
  point group; this agent opts in via ``SkillRegistry`` and the model calls
  them as ``self.web.search_web(...)`` / ``self.web.browse(url)``.
- ``research``    — agentic (CodeAct): the model writes Python that searches,
                    browses, and returns a validated ``ResearchReport``.
- ``today``       — state field (weekday, date, local time + UTC offset)
                    visible to the model for resolving relative dates.

Run:
    uv run xng-agent "topic to research"
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime

from nooa import Agent, strategy
from nooa.skill_registry import SkillRegistry
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm.registry import get_llm_client
from pydantic import BaseModel, Field

from xng_agent.skills import WebResearchSkill

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
    """You are a web research agent. Search the live web with
    self.web.search_web(), then open the most promising pages with
    self.web.browse(url) to read their real content, and synthesize a cited
    report. Browsed pages are recorded in self.web.findings."""

    today: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Visible in the model's state block: the model has no clock of its
        # own, so relative dates ("this weekend") must be resolved against
        # this. Weekday first: it is what relative-date resolution needs.
        # astimezone(): %z formats empty on naive datetimes, so attach the
        # local tzinfo explicitly.
        self.today = datetime.now().astimezone().strftime("%A %Y-%m-%d %H:%M %z")
        # Opt in to the web research skill: the entry point in pyproject.toml
        # ([project.entry-points."nooa.skills"]) maps "xng.web" to
        # WebResearchSkill; activate() loads it dynamically and exposes it to
        # the model as self.web.
        self.skills = SkillRegistry(self)
        self.web: WebResearchSkill
        self.skills.activate(["xng.web"])

    # --- Agentic method: ellipsis body, run by the LLM (CodeAct loop) ---

    # ty: the ellipsis body is intentional — NOOA implements it at runtime via the LLM.
    @strategy(CodeActStrategy())
    async def research(self, topic: str) -> ResearchReport:  # ty: ignore[empty-body]
        """Research {topic}. If the topic uses relative dates ("this weekend",
        "next month"), resolve them to concrete dates from self.today first and
        use those dates in your searches. Search the web with await
        self.web.search_web(), pick the 2-3 most promising URLs, and read each
        with await self.web.browse(url). Base the report only on the browsed
        content and search snippets — do not invent content. Cite a URL for
        each key fact."""
        ...


async def _run(topic: str) -> tuple[ResearchReport, int]:
    """Research *topic*, always killing the shared browser session afterwards."""
    agent = XngBrowserAgent()
    try:
        return await agent.research(topic), len(agent.web.findings)
    finally:
        await agent.web.aclose()


def main() -> None:
    """Run the agent on a topic from argv and print the validated report."""
    topic = " ".join(sys.argv[1:]) or "What is SearXNG and why do people self-host it?"
    report, browsed = asyncio.run(_run(topic))
    print("\nREPORT:")
    print(report.model_dump_json(indent=2))
    print(f"\nbrowsed {browsed} pages")


if __name__ == "__main__":
    main()
