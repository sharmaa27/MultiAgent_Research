"""Agent 2: Synthesizer — turns sources into a CITED answer.

═══ DECISIONS MADE IN THIS FILE (interview prep) ═══
D1. Sources go in the prompt as a numbered block, not raw JSON.
    Why: LLMs cite reliably against "[1] Title — text" format; raw JSON
    makes them cite keys or hallucinate ids.
D2. The system prompt FORBIDS outside knowledge and REQUIRES a citation
    per sentence. Why: without both rules the model blends its training
    data into the answer, and the Critic then can't verify anything.
D3. Explicit escape hatch: "if sources don't cover it, say so."
    Why: without a permitted way to say "I don't know", models fabricate.
    This one line is the single biggest hallucination reducer here.
D4. tier="strong" for synthesis. Why: writing quality + citation
    discipline degrade most on weak models. Research/critique survive
    on cheap models; synthesis doesn't. (Verify this in YOUR eval.)
D5. Revision = same prompt + critique injected, not a "fix it" chat.
    Why: stateless calls are reproducible and debuggable; multi-turn
    state is where agent bugs hide.
"""

from llm import llm_call
from logger import log_event

SYSTEM = """You are a research writer. Rules, in priority order:
1. Use ONLY the numbered sources provided. Your own knowledge is FORBIDDEN.
2. Every bullet or sentence with a factual claim ends with citation(s) like [1] or [1][3].
3. If the sources do not cover part of the question, write exactly:
   "The sources do not cover: <topic>." Do not guess.

FORMAT — a comprehensive research brief, structured for scanning:
- Start with a 2-3 sentence **bold** executive summary, cited.
- Then 4-6 sections with ## headings covering, where the sources allow:
  what it is / how it works / key components / trade-offs or comparisons /
  concrete numbers, examples, or use cases.
- Under each heading: bullet points (each 1-2 lines, each cited).
- **Bold** the key terms inside bullets.
- Be EXHAUSTIVE with the sources: extract every relevant fact, number,
  and example they contain. Aim for 500-900 words when the sources
  support it — but NEVER pad beyond what the sources actually say.
- If the topic has a process, flow, or structure, include ONE simple
  ASCII diagram inside a ```text code block``` (under 10 lines, built
  only from source information). Skip it if nothing structural to draw.
- End with a "## Gaps" section listing what the sources do NOT cover,
  so the reader knows the limits of this brief.
- No preamble, no fluff, no paragraph longer than 2 sentences."""

ANSWER_PROMPT = """Sources:
{source_block}

Question: {query}

Write a cited answer following your rules."""

REVISION_SUFFIX = """

A fact-checker reviewed your previous draft and flagged these problems:
{critique}

Previous draft:
{previous}

Rewrite the answer fixing every flagged problem. Same citation rules apply."""


def _format_sources(sources: list[dict]) -> str:
    # D1: numbered plain-text block. Truncate each source to ~1500 chars —
    # cost control; full pages blow up the context for marginal gain.
    return "\n\n".join(
        f"[{s['id']}] {s['title']}\n{s['content'][:1500]}" for s in sources
    )


def run_synthesizer(query: str, sources: list[dict],
                    critique: str | None = None,
                    previous: str | None = None) -> str:
    log_event("agent_start", {"agent": "synthesizer", "revision": critique is not None})

    prompt = ANSWER_PROMPT.format(source_block=_format_sources(sources), query=query)
    if critique:  # D5: revision is the same stateless call + injected feedback
        prompt += REVISION_SUFFIX.format(critique=critique, previous=previous or "")

    answer = llm_call(prompt, system=SYSTEM, tier="strong", max_tokens=2600)

    log_event("agent_end", {"agent": "synthesizer", "chars": len(answer)})
    return answer
