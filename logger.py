"""Structured JSON logging for every agent step.

Why this exists (interview answer): multi-agent failures cascade. Without
per-step traces you cannot tell WHICH agent went wrong. Every tool call,
LLM call, and loop iteration gets logged with a timestamp to runs/<id>.jsonl.
"""

import json
import os
import time
import uuid

RUN_ID = str(uuid.uuid4())[:8]
LOG_DIR = "runs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"{RUN_ID}.jsonl")


def log_event(event_type: str, payload: dict) -> None:
    record = {"ts": time.time(), "run_id": RUN_ID, "event": event_type, **payload}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def summarize_run() -> dict:
    """Aggregate token usage and step counts for the current run."""
    input_toks = output_toks = llm_calls = 0
    with open(LOG_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["event"] == "llm_call":
                llm_calls += 1
                input_toks += rec.get("input_tokens") or 0
                output_toks += rec.get("output_tokens") or 0
    return {
        "run_id": RUN_ID,
        "llm_calls": llm_calls,
        "input_tokens": input_toks,
        "output_tokens": output_toks,
    }
