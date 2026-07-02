"""Orchestrator: plain Python, no framework.

═══ DECISIONS ═══
D11. No LangGraph/CrewAI. A 3-agent pipeline is ~40 lines of control flow;
     a framework would hide exactly the state-passing and loop logic that
     interviews probe. v2 migration is the growth story.
D12. MAX_REVISION_LOOPS = 2, a HARD cap. Two agents can disagree forever
     (synthesizer fixes A, critic flags B, repeat). Without a cap: infinite
     token burn. Why 2 specifically? Run the eval at cap=1/2/3 — my
     hypothesis is loop 1 catches most issues and loop 3 adds cost, not
     accuracy. Put YOUR measured table in the README.
D13. critic_enabled flag exists so the eval can measure the critic's value
     (ON vs OFF). If you can't turn a component off, you can't measure it.
D14. Three distinct honesty states surfaced to the user:
     - verified (critic approved)
     - unverified (critic errored -> "could not be fact-checked")
     - contested (loop cap hit -> show the remaining issues verbatim)
     Never collapse these into one confident-looking answer.
"""

import sys
from agents.researcher import run_researcher
from agents.synthesizer import run_synthesizer
from agents.critic import run_critic
from logger import log_event, summarize_run

MAX_REVISION_LOOPS = 2  # D12


def answer_query(query: str, critic_enabled: bool = True) -> dict:
    # ---- Stage 1: research ----
    sources = run_researcher(query)
    if not sources:
        return {"answer": None,
                "error": "Search unavailable or returned nothing. Cannot answer reliably."}

    # ---- Stage 2/3: synthesize <-> critique loop ----
    critique = previous = None
    answer = ""
    for loop in range(MAX_REVISION_LOOPS + 1):
        answer = run_synthesizer(query, sources, critique=critique, previous=previous)

        if not critic_enabled:  # D13: eval baseline mode
            return {"answer": answer, "sources": sources, "loops_used": 0,
                    "caveats": "fact-checking disabled", "usage": summarize_run()}

        verdict = run_critic(answer, sources)
        log_event("loop", {"iteration": loop, "approved": verdict["approved"],
                           "critic_error": verdict["critic_error"],
                           "n_issues": len(verdict["issues"])})

        if verdict["critic_error"]:  # D14: unverified, not verified
            return {"answer": answer, "sources": sources, "loops_used": loop,
                    "caveats": "This answer could NOT be fact-checked (verifier failed).",
                    "usage": summarize_run()}

        if verdict["approved"]:      # D14: verified
            return {"answer": answer, "sources": sources, "loops_used": loop,
                    "caveats": None, "usage": summarize_run()}

        critique = "; ".join(verdict["issues"])
        previous = answer

    # D14: contested — loop cap hit with open issues
    return {"answer": answer, "sources": sources, "loops_used": MAX_REVISION_LOOPS,
            "caveats": f"Fact-checker still flags: {critique}",
            "usage": summarize_run()}


if __name__ == "__main__":
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or input("Research question: ")
    result = answer_query(q, critic_enabled="--no-critic" not in sys.argv)
    if result.get("error"):
        print(f"\n[FAILED] {result['error']}")
    else:
        print(f"\n{result['answer']}\n")
        if result["caveats"]:
            print(f"!! {result['caveats']}\n")
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['id']}] {s['title']} — {s['url']}")
        print(f"\nRun stats: {result['usage']}")
