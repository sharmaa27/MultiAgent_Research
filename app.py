"""Streamlit frontend v2 — richer UI for the multi-agent research assistant.

Features on top of v1:
- Example-question chips (one-click demo starters)
- Session history: every query this session, revisitable
- Sidebar: architecture summary, honesty-state legend, session totals
- Source cards with domain badges
- "How it works" tab explaining the pipeline (doubles as demo talking points)
- Planted-lie critic test (unchanged — D22)

Run:  streamlit run app.py
"""

import json
import time
from urllib.parse import urlparse

import streamlit as st

from orchestrator import answer_query, MAX_REVISION_LOOPS
from agents.critic import run_critic
import logger

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔎", layout="wide")

# ── styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 2.2rem; }
.hero {
  padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
  background: linear-gradient(120deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
  color: #fff;
}
.hero h1 { margin: 0 0 .3rem 0; font-size: 1.9rem; color: #fff; }
.hero p  { margin: 0; opacity: .85; }
.pipeline {
  font-family: monospace; background: rgba(255,255,255,.08);
  padding: .5rem .8rem; border-radius: 8px; display: inline-block; margin-top: .7rem;
}
.source-card {
  border: 1px solid rgba(128,128,128,.25); border-radius: 10px;
  padding: .7rem .9rem; margin-bottom: .6rem;
}
.source-card .domain {
  font-size: .75rem; opacity: .7; text-transform: uppercase; letter-spacing: .04em;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1> Multi-Agent Research Assistant</h1>
  <p>Three cooperating agents with a hallucination-catching critique loop.</p>
  <div class="pipeline">query → Researcher → Synthesizer ⇄ Critic → verified answer</div>
</div>
""", unsafe_allow_html=True)

# ── session state ────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []          # list of {query, result, elapsed, trace}
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

# ── sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader(" Pipeline")
    critic_on = st.toggle("Critic enabled", value=True,
                          help="Off = unverified baseline (what the eval compares against)")
    st.caption(f"Revision loop cap: **{MAX_REVISION_LOOPS}**")

    st.divider()
    st.subheader(" Verdict legend")
    st.markdown("""
- 🟢 **Verified** — critic approved every claim
- 🟡 **Unverified** — critic itself failed; honestly flagged
- 🔴 **Contested** — loop cap hit, objections shown
- ⛔ **Refused** — search failed, no fabrication
""")

    if st.session_state.history:
        st.divider()
        st.subheader("📊 Session totals")
        tot_tokens = sum(h["result"].get("usage", {}).get("input_tokens", 0)
                         + h["result"].get("usage", {}).get("output_tokens", 0)
                         for h in st.session_state.history if not h["result"].get("error"))
        st.metric("Queries", len(st.session_state.history))
        st.metric("Tokens used", f"{tot_tokens:,}")

# ── helpers ──────────────────────────────────────────────────────────
def read_trace_since(ts: float) -> list[dict]:
    events = []
    try:
        with open(logger.LOG_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec["ts"] >= ts:
                    events.append(rec)
    except FileNotFoundError:
        pass
    return events


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "") or "unknown"
    except Exception:
        return "unknown"


def render_result(query: str, result: dict, elapsed: float, trace: list[dict]):
    if result.get("error"):
        st.error(f"⛔ **Honest refusal:** {result['error']}")
        st.caption("No sources → no answer. The system never fabricates around missing data (D14).")
        return

    caveats = result["caveats"]
    if caveats is None:
        st.success("🟢 **Verified** — the critic checked every claim against its cited source")
    elif "could NOT be fact-checked" in caveats:
        st.warning(f"🟡 **Unverified** — {caveats}")
    elif "disabled" in caveats:
        st.info("⚪ **Critic disabled** — this is the unchecked baseline")
    else:
        st.error(f"🔴 **Contested** — {caveats}")

    st.markdown(result["answer"])
    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Sources")
        for s in result["sources"]:
            st.markdown(
                f"""<div class="source-card">
                <div class="domain">{domain_of(s['url'])}</div>
                <b>[{s['id']}]</b> <a href="{s['url']}" target="_blank">{s['title']}</a>
                </div>""",
                unsafe_allow_html=True,
            )
    with right:
        st.subheader("Run metrics")
        u = result["usage"]
        m1, m2 = st.columns(2)
        m1.metric("LLM calls", u["llm_calls"])
        m2.metric("Tokens", f"{u['input_tokens'] + u['output_tokens']:,}")
        m3, m4 = st.columns(2)
        m3.metric("Revision loops", result["loops_used"])
        m4.metric("Latency", f"{elapsed:.1f}s")

    with st.expander("🔬 Agent trace (every step, from the JSON log)"):
        for ev in trace:
            detail = {k: v for k, v in ev.items() if k not in ("ts", "run_id", "event")}
            st.code(f'{ev["event"]}: {json.dumps(detail, ensure_ascii=False)}', language="json")


# ── tabs ─────────────────────────────────────────────────────────────
tab_ask, tab_how, tab_test = st.tabs([" Ask", " How it works", " Test the Critic"])

with tab_ask:
    st.markdown("**Try an example:**")
    examples = [
        "What are the trade-offs between RAG and fine-tuning?",
        "How do transformer attention mechanisms work?",
        "Is nuclear energy cost-competitive with solar in 2026?",
    ]
    chip_cols = st.columns(len(examples))
    for col, ex in zip(chip_cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state.pending_query = ex

    query = st.text_input("Research question",
                          value=st.session_state.pending_query,
                          placeholder="Ask anything that needs sourced research...")

    if st.button("Research", type="primary", disabled=not query):
        st.session_state.pending_query = ""
        t0 = time.time()
        try:
            with st.spinner("Researcher searching → Synthesizer writing → Critic checking..."):
                result = answer_query(query, critic_enabled=critic_on)
        except Exception as e:
            # Last-resort catch: show the reason readably instead of a raw
            # traceback. The informative errors from llm.py land here.
            result = {"error": f"Pipeline failure — {e}"}
        elapsed = time.time() - t0
        trace = read_trace_since(t0)
        st.session_state.history.append(
            {"query": query, "result": result, "elapsed": elapsed, "trace": trace})

    if st.session_state.history:
        latest = st.session_state.history[-1]
        render_result(latest["query"], latest["result"], latest["elapsed"], latest["trace"])

        if len(st.session_state.history) > 1:
            st.divider()
            st.subheader("🕘 Earlier this session")
            for h in reversed(st.session_state.history[:-1]):
                with st.expander(f"“{h['query']}”"):
                    render_result(h["query"], h["result"], h["elapsed"], h["trace"])

with tab_how:
    st.subheader("The pipeline")
    st.code("""query
  └─> RESEARCHER   breaks it into sub-questions, searches the web (Tavily),
  │                dedupes results into numbered sources
  └─> SYNTHESIZER  writes an answer using ONLY those sources,
  │                citing [n] after every factual claim
  └─> CRITIC       verifies each claim against its cited source
        ├─ approved  → 🟢 verified answer
        ├─ rejected  → back to Synthesizer with the issues (max 2 loops)
        │              still rejected at cap → 🔴 contested, issues shown
        └─ critic crashed → 🟡 answer shown but marked "not fact-checked"
""", language=None)
    st.subheader("Why it's built this way")
    st.markdown("""
- **Plain-Python orchestration, no agent framework** — the loop control and state
  passing stay visible and debuggable (D11).
- **Hard cap of 2 revision loops** — two models can disagree forever; the cap is
  a circuit breaker against infinite token burn (D12).
- **Fail-honest critic** — if verification itself breaks, the answer is labeled
  *unverified* rather than silently trusted or fake-rejected (D9).
- **Cheap model for critique, strong for synthesis** — verification is easier
  than generation; the eval measures whether that holds (D7).
- **429-aware retry** — rate limits are retried with the server's own
  `retry-after` hint; auth errors fail immediately (retryable vs not).
""")

with tab_test:
    st.markdown("""**The planted-lie test (D10):** a critic that approves everything looks
identical to a working one on good answers. This feeds it a claim that
**contradicts** its source — a working critic must reject it.""")
    st.code('Source [1] says: "The sky is blue."\nAnswer claims:  "The sky is green. [1]"')
    if st.button("Run planted-lie test"):
        src = [{"id": 1, "title": "Doc", "url": "x", "content": "The sky is blue."}]
        with st.spinner("Critic checking..."):
            verdict = run_critic("The sky is green. [1]", src)
        if verdict["critic_error"]:
            st.warning("Critic output unparseable — fail-honest path triggered (D9). Rerun once.")
        elif verdict["approved"]:
            st.error("❌ FAILED: the critic approved a lie — the prompt in agents/critic.py needs tightening.")
        else:
            st.success("✅ PASSED: the critic rejected the fabricated claim.")
            for issue in verdict["issues"]:
                st.markdown(f"- {issue}")
