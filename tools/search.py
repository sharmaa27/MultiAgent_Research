"""Web search tool (Tavily — free tier at tavily.com).

Failure handling is deliberate and visible: retries with backoff, then a
structured empty result instead of a crash. Interview question this answers:
"What happens when your search tool fails?" -> agent surfaces uncertainty
instead of hallucinating around missing data.
"""

import os
import time
import requests
from logger import log_event

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")

MAX_RETRIES = 2


def search(query: str, max_results: int = 5) -> list[dict]:
    """Returns list of {title, url, content}. Empty list = search failed (callers must handle)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                TAVILY_URL,
                json={"api_key": TAVILY_KEY, "query": query, "max_results": max_results},
                timeout=20,
            )
            resp.raise_for_status()
            results = [
                {"title": r["title"], "url": r["url"], "content": r["content"]}
                for r in resp.json().get("results", [])
            ]
            log_event("tool_call", {"tool": "search", "query": query, "n_results": len(results)})
            return results
        except requests.RequestException as e:
            log_event("tool_error", {"tool": "search", "attempt": attempt, "error": str(e)})
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s
    return []  # graceful failure — orchestrator decides what to do
