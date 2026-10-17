"""Real OpenAI connector for Part 1. Returns a function  llm(prompt) -> str."""
import logging
import os
import time

log = logging.getLogger("support.llm")


def make_openai_llm(model: str | None = None):
    from openai import OpenAI  # imported lazily so offline tests don't need a key

    client = OpenAI(timeout=30, max_retries=3)       # SDK retries 429/5xx with backoff
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def llm(prompt: str) -> str:
        # The pipeline's classify / gate prompts expect JSON -> use JSON mode for those.
        wants_json = prompt.startswith(("Classify", "Answer JSON"))
        kwargs = {"response_format": {"type": "json_object"}} if wants_json else {}
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=model, temperature=0, messages=[{"role": "user", "content": prompt}], **kwargs)
        log.info("openai call %.1fs tokens=%s", time.perf_counter() - t0, r.usage.total_tokens if r.usage else "?")
        return r.choices[0].message.content or ""

    return llm
