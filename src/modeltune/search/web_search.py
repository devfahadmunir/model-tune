"""
web_search.py

Minimal, dependency-light web search used to ground model answers in current
information. Deliberately returns short text snippets (title + body), not
full page content — good enough to answer "what's happening with X" style
questions, and much faster/simpler than crawling full pages.

This module has no knowledge of chat formatting or the model at all — it
just answers one question: "given a query, what does the web say?" Keeping
it decoupled like this means it's reusable later as an MCP tool, a batch
research script, or anything else, without depending on the serving gateway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("modeltune.search")


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Run a web search and return a short list of results.

    Raises no exception on an empty result set — callers should check for
    an empty list and decide how to handle "nothing found" themselves,
    rather than this function guessing what that should mean.
    """
    try:
        from ddgs import DDGS
    except ImportError as e:
        raise RuntimeError(
            "The 'ddgs' package is required for web search. Install with: pip install ddgs"
        ) from e

    results: list[SearchResult] = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=max_results):
            results.append(
                SearchResult(
                    title=hit.get("title", ""),
                    snippet=hit.get("body", ""),
                    url=hit.get("href", ""),
                )
            )

    logger.info("Search for %r returned %d result(s).", query, len(results))
    return results


def format_results_for_prompt(results: list[SearchResult]) -> str:
    """
    Turn search results into a compact block of text suitable for inserting
    into a model prompt as reference context. Kept deliberately plain (no
    markdown tables, no nested structure) since that's what small models
    tend to follow most reliably.
    """
    if not results:
        return "No web search results were found for this query."

    lines = ["Web search results:"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r.title}\n{r.snippet}\nSource: {r.url}")
    return "\n".join(lines)