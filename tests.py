"""Tests that run WITHOUT API keys, using mocks (monkeypatching).

═══ DECISIONS ═══
D18. Mock the LLM and search, test the ORCHESTRATION. Why: agent logic
     bugs (loop control, state passing, failure paths) are deterministic
     and testable; LLM outputs aren't. Separate the two.
D19. Test the failure paths explicitly: search-down, critic rejection loop,
     loop-cap hit, critic malformed output. The happy path is the least
     interesting test in an agent system.

Run: python tests.py
"""

import json
import sys
import types

# ---- Build fake modules BEFORE importing the real code ----
calls = {"synth": 0, "critic": 0}

fake_llm = types.ModuleType("llm")
def _fake_llm_call(prompt, system="", tier="cheap", max_tokens=1500):
    return "unused"
fake_llm.llm_call = _fake_llm_call
sys.modules["llm"] = fake_llm

import orchestrator  # noqa: E402  (imports agents, which import fake llm)
from agents import researcher, synthesizer, critic  # noqa: E402


SOURCES = [{"id": 1, "title": "Doc", "url": "http://x", "content": "The sky is blue."}]


def test_search_failure_is_surfaced():
    orchestrator.run_researcher = lambda q: []
    # Re-bind inside orchestrator's namespace
    result = orchestrator.answer_query("anything")
    assert result["answer"] is None and "Cannot answer reliably" in result["error"]
    print("PASS: search failure -> honest error, no fabricated answer")


def test_revision_loop_and_cap():
    orchestrator.run_researcher = lambda q: SOURCES
    orchestrator.run_synthesizer = lambda q, s, critique=None, previous=None: "The sky is green. [1]"
    # Critic ALWAYS rejects -> loop must stop at MAX_REVISION_LOOPS and surface issues
    orchestrator.run_critic = lambda a, s: {"approved": False, "critic_error": False,
                                            "issues": ["sky color contradicts source [1]"]}
    result = orchestrator.answer_query("what color is the sky")
    assert result["loops_used"] == orchestrator.MAX_REVISION_LOOPS
    assert "still flags" in result["caveats"]
    print("PASS: infinite-disagreement stopped by loop cap, issues surfaced verbatim")


def test_critic_error_means_unverified():
    orchestrator.run_researcher = lambda q: SOURCES
    orchestrator.run_synthesizer = lambda q, s, critique=None, previous=None: "The sky is blue. [1]"
    orchestrator.run_critic = lambda a, s: {"approved": True, "critic_error": True, "issues": []}
    result = orchestrator.answer_query("q")
    assert "could NOT be fact-checked" in result["caveats"]
    print("PASS: critic failure -> answer marked unverified, not silently trusted")


def test_approved_answer_passes_clean():
    orchestrator.run_researcher = lambda q: SOURCES
    orchestrator.run_synthesizer = lambda q, s, critique=None, previous=None: "The sky is blue. [1]"
    orchestrator.run_critic = lambda a, s: {"approved": True, "critic_error": False, "issues": []}
    result = orchestrator.answer_query("q")
    assert result["caveats"] is None and result["loops_used"] == 0
    print("PASS: verified answer returns with zero caveats")


def test_critic_json_parsing():
    good = critic._parse_verdict('{"approved": false, "issues": ["x"]}')
    fenced = critic._parse_verdict('```json\n{"approved": true, "issues": []}\n```')
    garbage = critic._parse_verdict('sure! the answer looks great to me')
    assert good == {"approved": False, "issues": ["x"]}
    assert fenced == {"approved": True, "issues": []}
    assert garbage is None
    print("PASS: verdict parser handles clean/fenced/garbage LLM output")


# NOTE (D10): the LIVE fabrication test — plant "The sky is green [1]" against
# a source saying blue, with real API keys, and confirm the real Critic
# rejects it. Run manually once keys are set:
#   from agents.critic import run_critic
#   run_critic("The sky is green. [1]", SOURCES)  -> approved must be False

if __name__ == "__main__":
    test_search_failure_is_surfaced()
    test_revision_loop_and_cap()
    test_critic_error_means_unverified()
    test_approved_answer_passes_clean()
    test_critic_json_parsing()
    print("\nAll orchestration tests passed (no API keys needed).")
