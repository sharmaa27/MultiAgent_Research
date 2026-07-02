# Multi-Agent Research Assistant

Answers research questions using three cooperating agents, with a critique
loop that reduces hallucinations — and a test suite proving the failure paths.

```
query ──> [Researcher] ──sources──> [Synthesizer] ──answer──> [Critic]
              │                          ▲                       │
          web search                     └──── issues (max 2 ────┘
          (Tavily)                             revision loops)
```

## Run

```bash
pip install requests
export ANTHROPIC_API_KEY=...   # or swap provider in llm.py
export TAVILY_API_KEY=...      # free at tavily.com
python tests.py                             # no keys needed — orchestration tests
python orchestrator.py "your question"      # live run
python eval/eval.py && python eval/eval.py --no-critic   # the ON/OFF comparison
```

## Design decisions (D1–D19)

Every decision is numbered and explained as a comment block at the top of the
file where it lives: llm.py, agents/*.py, orchestrator.py, eval/eval.py, tests.py.
Highlights:

- **D3** — the "sources don't cover this" escape hatch is the single biggest
  hallucination reducer.
- **D9** — critic failure = "unverified", never silently approved or fake-rejected.
- **D12** — revision loop hard-capped at 2 (measure cap=1/2/3 in the eval).
- **D14** — three honesty states: verified / unverified / contested.
- **D17** — the eval judge is independent of the critic, or the eval is circular.

## Eval results (fill in after running)

| Config | Claims supported | Total tokens | Notes |
|---|---|---|---|
| Critic OFF | __% | __ | |
| Critic ON  | __% | __ | |

## TODO (v2)
- Write remaining 19 eval questions (incl. 2-3 unanswerable ones)
- Migrate orchestrator to LangGraph; compare code complexity
- Streamlit UI + deploy
