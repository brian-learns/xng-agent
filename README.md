# xng-agent

A [NOOA](https://arxiv.org/abs/2607.20709) agent that researches a topic using live web
search (`xng` / SearXNG) and real page browsing (`browser-use` primitives). No memory, no
browser sub-agent: the only LLM loop is NOOA's.

The agent is a single Python class (`src/xng_agent/__init__.py`); the model-facing surface:

- `await search_web(query, limit)` — deterministic; wraps the `xng` Python API in a worker
  thread (`asyncio.to_thread`) so the blocking HTTP call can't stall the browser loop.
- `browse(url)` — deterministic; opens the URL in a shared headless Chrome and returns the
  page's visible text (truncated to 3000 chars) and title. Drives browser-use session
  primitives directly (CDP navigate + `document.body.innerText`), so no nested LLM loop.
- `research(topic)` — agentic (CodeAct, `...` body): the model writes Python, searches,
  browses, and returns a validated `ResearchReport` (summary, cited key facts, sources).
- `today` — state field (weekday, date, local time + UTC offset) visible to the model so it can
  resolve relative dates like "this weekend" to concrete dates.

## Prerequisites

- A `llama-server` router (OpenAI-compatible). Defaults: `http://127.0.0.1:8080/v1` with
  model `Qwen3.6-35B-A3B-MXFP4_MOE`; override with `NOOA_MODEL` / `NOOA_LLM_BASE`.
- A running SearXNG instance (configured for `xng`, e.g. `SEARXNG_URL`).
- A local Chrome at `/usr/bin/google-chrome` (override with `CHROME_PATH`).

## Usage

Starting the nooa dev console first and leaving it open makes it easy to review the agent traces.

```
export NOOA_VIEWER_AUTH_TOKEN=$(openssl rand -hex 16)
uv run nooa start-dev -h 0.0.0.0
```

Then run the agent

```
uv run xng-agent "subject to search"
```

## Notes

- One shared Chrome session per run: started lazily, and `browse()` waits until the CDP
  connection is actually open before touching a page (browser-use's `start()` returns before
  the launch handler finishes). Every page operation is timeout-bounded; a hung or crashed
  session is killed and replaced on the next call.
- The session is torn down with `kill()` at the end of the run. browser-use's `stop()`
  deliberately keeps the browser process alive (for reconnecting) — using it leaks Chrome.
- `browse()` returns the **final** URL after redirects and reports empty content honestly
  for pages that only render via JS; the model is expected to skip those.
- Thinking is disabled for the NOOA loop via `chat_template_kwargs` (litellm `extra_body`)
  for speed.
