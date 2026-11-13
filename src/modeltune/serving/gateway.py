"""
gateway.py

A thin OpenAI-compatible proxy that sits between a chat client (Open WebUI,
or anything else) and one of several model backends (local mlx_lm.server,
Gemini, OpenAI, Claude), selected per-request via the normal OpenAI
"model" field.

Responsibilities layered on top of the raw backends, kept in this one
place rather than duplicated per client:

1. BASE_SYSTEM_PROMPT — a persistent identity/instructions message
   prepended to every real conversation.
2. Search augmentation — an explicit "/search " prefix or an automatic
   keyword heuristic triggers a real web search, injected as context.
3. Tool-stripping for the "local" backend only — mlx_lm.server cannot
   safely handle OpenAI-style tool/function-calling (see the note below).
4. Skips Open WebUI's own internal background requests (title/tag/
   follow-up generation) entirely, since augmenting those corrupts them.

Run with:
    uvicorn modeltune.serving.gateway:app --host 0.0.0.0 --port 8100
or:
    modeltune-gateway   (after `uv pip install -e .`)

Configuration is read from environment variables, loaded automatically
from a .env file in the project root if present (see .env.example).
"""

from __future__ import annotations

import logging
import os
import re

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from modeltune.search.web_search import format_results_for_prompt, search_web

# Loads variables from a .env file in the current working directory (or
# any parent) into os.environ, if one exists. Safe to call even when no
# .env file is present — it just does nothing in that case. Real secrets
# (API keys) should live in .env, never hardcoded in this file.
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("modeltune.gateway")

MLX_SERVER_URL = os.environ.get("MLX_SERVER_URL", "http://localhost:8099/v1")
SEARCH_TRIGGER = "/search "

# --- Model routing --------------------------------------------------------
# "api_key_env" holds the NAME of an environment variable, never the key
# itself — the actual value is looked up at request time in
# build_headers(). This keeps real secrets out of this file entirely;
# they only ever need to exist in .env (or the real environment).
DEFAULT_MODEL = "fused-model"

MODEL_ROUTES = {
    "fused-model": {
        "url": MLX_SERVER_URL,
        "type": "local",
    },
    "gemini-3.5-flash-lite": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "type": "openai-compatible",
        "api_key_env": "GEMINI_API_KEY",
    },
    "gpt-4o": {
        "url": "https://api.openai.com/v1",
        "type": "openai-compatible",
        "api_key_env": "OPENAI_API_KEY",
    },
    "claude-sonnet-4-5": {
        "url": "https://api.anthropic.com/v1",
        "type": "openai-compatible",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
}


def resolve_route(model_name: str) -> tuple[str, dict]:
    """
    Look up the requested model name in MODEL_ROUTES, falling back to
    DEFAULT_MODEL (with a warning) for anything unrecognized — so a typo'd
    or unconfigured model name fails safe to the local model rather than
    crashing the request outright.
    """
    route = MODEL_ROUTES.get(model_name)
    if route is None:
        log.warning("Unknown model %r requested — falling back to %r.", model_name, DEFAULT_MODEL)
        model_name = DEFAULT_MODEL
        route = MODEL_ROUTES[DEFAULT_MODEL]
    return model_name, route


def build_headers(route: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    api_key_env = route.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set in the environment (.env or real env var).")
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# --- Agent identity -----------------------------------------------------
BASE_SYSTEM_PROMPT = (
    "Your name is Lisbon. You are a helpful assistant that helps the user "
    "with their daily tasks. Be concise, direct, and honest — if you don't "
    "know something or don't have access to real-time information, say so "
    "clearly instead of guessing."
)

# --- Auto-search heuristic ----------------------------------------------
AUTO_SEARCH_PATTERNS = re.compile(
    r"\b("
    r"today|currently|current|latest|right now|this week|"
    r"weather|forecast|news|headlines|"
    r"price of|stock price|exchange rate|"
    r"score|match result|"
    r"who is the (current|new)|what time is it"
    r")\b",
    re.IGNORECASE,
)

app = FastAPI(title="modeltune gateway")


def needs_auto_search(text: str) -> bool:
    return bool(AUTO_SEARCH_PATTERNS.search(text))


def is_internal_task_request(messages: list[dict]) -> bool:
    """
    Open WebUI silently calls the chat endpoint for its own background
    features — title generation, tag generation, suggested follow-ups —
    using a much longer meta-prompt than anything the user actually typed.
    These always contain a literal "### Task:" marker. Augmenting these
    with search or the agent's system prompt wastes search quota on junk
    queries and can corrupt a request Open WebUI expects an exact format
    back from, so they're detected and passed through completely untouched.
    """
    return any("### Task:" in m.get("content", "") for m in messages)


def ensure_system_prompt(messages: list[dict]) -> list[dict]:
    if any(m.get("role") == "system" for m in messages):
        return messages
    return [{"role": "system", "content": BASE_SYSTEM_PROMPT}] + list(messages)


def maybe_augment_with_search(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages

    last = messages[-1]
    if last.get("role") != "user":
        return messages

    content = last.get("content", "")
    explicit = content.startswith(SEARCH_TRIGGER)
    auto = (not explicit) and needs_auto_search(content)

    if not (explicit or auto):
        return messages

    query = content[len(SEARCH_TRIGGER):].strip() if explicit else content
    trigger_kind = "explicit" if explicit else "auto"
    log.info("Search triggered (%s). Query: %r", trigger_kind, query)

    try:
        results = search_web(query, max_results=5)
        context_block = format_results_for_prompt(results)
    except Exception as e:
        log.error("Web search failed: %s", e)
        context_block = f"(Web search failed: {e})"

    augmented = list(messages[:-1])
    augmented.append({
        "role": "system",
        "content": (
            "The user asked a question that may require current information. "
            "Use the following web search results to answer accurately. "
            "If the results don't fully answer the question, say so honestly "
            "rather than guessing.\n\n" + context_block
        ),
    })
    augmented.append({"role": "user", "content": query})
    return augmented


async def stream_from_backend(url: str, headers: dict, body: dict):
    """
    Relay the upstream SSE stream one COMPLETE event at a time, not as
    arbitrary raw byte chunks. aiter_bytes() yields whatever happens to
    arrive off the network at a given moment — which can split a single
    "data: {...}\\n\\n" event across two chunks purely due to timing, with
    nothing wrong with the content itself. A client's incremental SSE
    parser expects each chunk to be a complete event; a split event can
    silently fail to render live even though the full response is fine
    once fully received (exactly the "works after refresh" symptom).
    aiter_lines() reassembles complete lines regardless of how the
    underlying TCP packets were chunked, so each yielded event is whole.
    """
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                yield f"{line}\n\n".encode("utf-8")


@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [{"id": name, "object": "model"} for name in MODEL_ROUTES],
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    model_name, route = resolve_route(body.get("model", DEFAULT_MODEL))
    body["model"] = model_name

    if route["type"] == "local" and ("tools" in body or "tool_choice" in body):
        log.warning("Stripping 'tools'/'tool_choice' — unsupported by the local backend (mlx_lm.server).")
        body.pop("tools", None)
        body.pop("tool_choice", None)

    if is_internal_task_request(messages):
        log.info("Internal Open WebUI task request detected (### Task:) — passing through untouched.")
    else:
        messages = maybe_augment_with_search(messages)
        messages = ensure_system_prompt(messages)
    body["messages"] = messages

    try:
        headers = build_headers(route)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    url = f"{route['url'].rstrip('/')}/chat/completions"

    if body.get("stream"):
        return StreamingResponse(
            stream_from_backend(url, headers, body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # tells any reverse proxy not to buffer the stream
                "Connection": "keep-alive",
            },
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, headers=headers, json=body)

    if resp.status_code >= 400:
        log.error("Backend %s returned %d: %s", route["url"], resp.status_code, resp.text)

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


def main() -> None:
    import uvicorn
    uvicorn.run("modeltune.serving.gateway:app", host="0.0.0.0", port=8100, reload=False)


if __name__ == "__main__":
    main()