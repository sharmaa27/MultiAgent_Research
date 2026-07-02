"""Agent 1: Researcher (REFERENCE IMPLEMENTATION — study this pattern,
then write Synthesizer and Critic yourself following the same shape).

Job: query -> sub-questions -> search each -> deduplicated source list.
"""

import json
from llm import llm_call
from tools.search import search
from logger import log_event

SUBQUESTION_PROMPT = """Break this research question into 2-4 specific,
searchable sub-questions. Respond with ONLY a JSON array of strings,
no markdown, no explanation.

Question: {query}"""


def run_researcher(query: str) -> list[dict]:
    """Returns numbered sources: [{id, title, url, content}, ...]"""
    log_event("agent_start", {"agent": "researcher", "query": query})

    # Step 1: decompose the query (cheap model — it's a simple task)
    raw = llm_call(SUBQUESTION_PROMPT.format(query=query), tier="cheap")
    try:
        sub_questions = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        # LLM returned malformed JSON — fall back to the original query.
        # (Interview point: never trust LLM output format blindly.)
        log_event("parse_error", {"agent": "researcher", "raw": raw[:200]})
        sub_questions = [query]

    # Step 2: search each sub-question, dedupe by URL, drop low-quality domains.
    # Deny-list motivated by three logged incidents: a Facebook group post and
    # two YouTube videos surfaced as "sources"; the critic even flagged a claim
    # as supported only by a video description. Social/video pages aren't
    # verifiable text sources for a fact-checking pipeline.
    DENY_DOMAINS = ("youtube.com", "youtu.be", "facebook.com", "reddit.com",
                    "twitter.com", "x.com", "instagram.com", "tiktok.com",
                    "quora.com", "pinterest.com")
    seen_urls: set[str] = set()
    sources: list[dict] = []
    for sq in sub_questions[:4]:  # hard cap — cost control
        for r in search(sq, max_results=4):
            if any(d in r["url"] for d in DENY_DOMAINS):
                log_event("source_filtered", {"url": r["url"]})
                continue
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                sources.append({"id": len(sources) + 1, **r})

    log_event("agent_end", {"agent": "researcher", "n_sources": len(sources)})
    return sources
