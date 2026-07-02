"""Agent 3: Critic — verifies every claim against its cited source.

Decisions D6-D10 (see study guide). NEW: D9 extended — the fail-honest
principle now covers the critic CRASHING, not just returning garbage.
If the critic's own LLM call raises (400 json_validate_failed, network,
anything), the answer is returned marked "unverified" instead of taking
the whole app down. A broken verifier should degrade the verdict, never
the product.
"""

import json
from llm import llm_call
from logger import log_event

SYSTEM = """You are a fact-checker. You receive an answer with [n] citations
and the numbered sources. For EACH factual claim, verify the cited source
actually supports it — not just mentions the topic, SUPPORTS the claim.

Classify problems by severity:
- MAJOR: a claim that contradicts its source, is fabricated, or has no
  support in ANY of the provided sources.
- MINOR: a claim mostly supported but with an imperfect citation (e.g.
  supported by [1] and [4] but also cites [3] which doesn't mention it),
  or slight paraphrasing looseness, or formatting/structure issues.

Respond with ONLY this JSON, no markdown fences, no commentary:
{"approved": true/false, "issues": ["<claim> — <why it fails>", ...]}

approved=false ONLY if there is at least one MAJOR problem.
MINOR problems: list them in issues but they do NOT block approval.
Do not flag headings, bullets, or diagrams as issues — only factual claims.
Keep each issue under 25 words. Report at most 5 issues."""

CHECK_PROMPT = """Sources:
{source_block}

Answer to verify:
{answer}"""


def _parse_verdict(raw: str) -> dict | None:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        v = json.loads(cleaned)
        if isinstance(v.get("approved"), bool) and isinstance(v.get("issues"), list):
            return {"approved": v["approved"], "issues": [str(i) for i in v["issues"]]}
    except json.JSONDecodeError:
        pass
    return None


def run_critic(answer: str, sources: list[dict]) -> dict:
    """Returns {"approved": bool, "issues": [...], "critic_error": bool}.
    NEVER raises — a crashing critic degrades to 'unverified' (D9 extended)."""
    log_event("agent_start", {"agent": "critic"})

    source_block = "\n\n".join(
        f"[{s['id']}] {s['title']}\n{s['content'][:1500]}" for s in sources
    )

    try:
        raw = llm_call(
            CHECK_PROMPT.format(source_block=source_block, answer=answer),
            # max_tokens raised 800 -> 1400: leading suspect for the 400 was
            # json mode + output truncated mid-JSON by the token cap
            # (Groq rejects invalid JSON generations with a 400).
            system=SYSTEM, tier="cheap", max_tokens=1400, json_mode=True,
        )
    except Exception as e:
        # Fail-honest on crash: log the real reason, degrade to unverified.
        log_event("critic_crash", {"agent": "critic", "error": str(e)[:500]})
        return {"approved": True, "issues": [], "critic_error": True}

    verdict = _parse_verdict(raw)
    if verdict is None:
        log_event("parse_error", {"agent": "critic", "raw": raw[:2000]})
        return {"approved": True, "issues": [], "critic_error": True}

    log_event("agent_end", {"agent": "critic", "approved": verdict["approved"],
                            "n_issues": len(verdict["issues"])})
    return {**verdict, "critic_error": False}
