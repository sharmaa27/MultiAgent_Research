"""Single place to swap LLM providers. Everything else calls llm_call().

Currently: Groq (OpenAI-compatible API, free tier). Originally Anthropic.
Includes: 429 retry with retry-after, informative errors (status + body).
"""

import os
import time
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.environ.get("GROQ_API_KEY", "")

MODELS = {
    "cheap": "llama-3.1-8b-instant",
    "strong": "llama-3.3-70b-versatile",
}


def llm_call(prompt: str, system: str = "", tier: str = "cheap",
             max_tokens: int = 1500, json_mode: bool = False) -> str:
    """Call the LLM and return plain text. Retries 429s; raises informative errors otherwise."""
    if not API_KEY:
        raise RuntimeError("Set GROQ_API_KEY env var (or swap provider in llm.py)")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Retry ONLY 429 (temporary: limit resets). 401/400 never fix themselves.
    for attempt in range(3):
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "content-type": "application/json",
            },
            json={
                "model": MODELS[tier],
                "max_tokens": max_tokens,
                "messages": messages,
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
            },
            timeout=60,
        )
        if resp.status_code == 429 and attempt < 2:
            wait = int(resp.headers.get("retry-after", 2 ** attempt + 1))
            time.sleep(wait)
            continue
        break

    # Informative errors: a bare status code hides the reason; the body has it.
    if not resp.ok:
        raise RuntimeError(f"Groq {resp.status_code}: {resp.text[:500]}")
    data = resp.json()

    usage = data.get("usage", {})
    from logger import log_event
    log_event("llm_call", {
        "tier": tier,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "finish_reason": data["choices"][0].get("finish_reason"),
    })
    return data["choices"][0]["message"]["content"]
