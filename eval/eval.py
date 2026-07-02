"""Eval harness — produces the critic ON vs OFF table (your resume bullet).

═══ DECISIONS ═══
D15. Metric = % of factual claims supported by cited sources, graded by an
     LLM judge. Why not exact-match accuracy? Research answers are
     free-form; there's no single right string to match.
D16. LLM-as-judge caveat: the judge can be wrong too. Mitigation: grade
     the first 5 answers MANUALLY, check the judge agrees with you (>80%
     agreement = trust it for the rest). Say this in interviews — knowing
     the limits of LLM-as-judge is senior-level awareness.
D17. Judge uses tier="strong" and a DIFFERENT prompt than the critic.
     If the judge were the same model+prompt as the critic, the eval would
     just measure the critic agreeing with itself. Independence matters.

Run:  python eval.py            (critic ON)
      python eval.py --no-critic (baseline)
Then compare results/on.json vs results/off.json.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator import answer_query  # noqa: E402
from llm import llm_call               # noqa: E402

JUDGE_SYSTEM = """You grade a research answer against its sources.
Count: (1) total factual claims, (2) claims fully supported by their cited source.
Respond ONLY with JSON: {"total_claims": N, "supported_claims": N}"""

JUDGE_PROMPT = """Sources:
{sources}

Answer:
{answer}"""


def judge(answer: str, sources: list[dict]) -> dict:
    source_block = "\n\n".join(f"[{s['id']}] {s['title']}\n{s['content'][:1500]}" for s in sources)
    raw = llm_call(JUDGE_PROMPT.format(sources=source_block, answer=answer),
                   system=JUDGE_SYSTEM, tier="strong", max_tokens=100)
    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"total_claims": 0, "supported_claims": 0, "judge_error": True}


def run_eval(critic_enabled: bool) -> None:
    with open(os.path.join(os.path.dirname(__file__), "questions.json")) as f:
        questions = json.load(f)

    os.makedirs("results", exist_ok=True)
    rows, total, supported, cost_tokens = [], 0, 0, 0

    for q in questions:
        t0 = time.time()
        res = answer_query(q["question"], critic_enabled=critic_enabled)
        latency = time.time() - t0
        if res.get("error"):
            rows.append({"id": q["id"], "error": res["error"]})
            continue
        g = judge(res["answer"], res["sources"])
        total += g.get("total_claims", 0)
        supported += g.get("supported_claims", 0)
        u = res["usage"]
        cost_tokens += u["input_tokens"] + u["output_tokens"]
        rows.append({"id": q["id"], "question": q["question"], "answer": res["answer"],
                     "grade": g, "latency_s": round(latency, 1),
                     "loops_used": res["loops_used"], "caveats": res["caveats"]})
        print(f"Q{q['id']}: {g.get('supported_claims')}/{g.get('total_claims')} "
              f"supported, {latency:.0f}s, loops={res['loops_used']}")

    pct = 100 * supported / total if total else 0
    mode = "on" if critic_enabled else "off"
    summary = {"critic": mode, "supported_pct": round(pct, 1),
               "total_claims": total, "total_tokens": cost_tokens}
    with open(f"results/{mode}.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"\n=== critic {mode.upper()}: {pct:.1f}% claims supported, "
          f"{cost_tokens} tokens total ===")
    print(f"Saved results/{mode}.json — grade the first 5 manually to validate the judge (D16).")


if __name__ == "__main__":
    run_eval(critic_enabled="--no-critic" not in sys.argv)
