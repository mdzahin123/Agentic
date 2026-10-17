"""Robust web summarizer.

Fixes over original_script.py:
  * resilient fetch  : timeouts, retries w/ backoff, UA header, content-type + size limits
  * real extraction  : trafilatura (fallback BeautifulSoup) strips boilerplate; Playwright
                       fallback renders JS-heavy pages when the static HTML has no content
  * long content     : paragraph-aware chunking + map-reduce summarization
  * output guardrail : hard word budget, validated; re-prompt with feedback, then deterministic
                       sentence-boundary truncation so the contract ALWAYS holds

Usage:  python fixed_summarizer.py https://example.com --max-words 120
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from typing import Callable

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("summarizer")
LLM = Callable[[str], str]

MAX_BYTES = 5_000_000            # refuse giant downloads
MIN_USEFUL_CHARS = 300           # below this the static page is probably a JS shell
CHUNK_CHARS = 12_000             # ~3k tokens per chunk: safe for any modern model
MAX_CHUNKS = 40                  # cost cap; beyond this we sample evenly


# ------------------------------------------------------------------ fetching
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "Mozilla/5.0 (compatible; SummarizerBot/1.0)"
    return s


def fetch_static(url: str, timeout: tuple[int, int] = (5, 20)) -> str:
    with _session().get(url, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        if "html" not in r.headers.get("Content-Type", "html"):
            raise ValueError(f"Unsupported content-type: {r.headers.get('Content-Type')}")
        buf, size = [], 0
        for part in r.iter_content(65_536):
            size += len(part)
            if size > MAX_BYTES:
                log.warning("page exceeded %d bytes; truncating download", MAX_BYTES)
                break
            buf.append(part)
        return b"".join(buf).decode(r.encoding or "utf-8", errors="replace")


def fetch_rendered(url: str, timeout_ms: int = 30_000) -> str:
    """Headless-browser fallback for JS-rendered pages (optional dependency)."""
    from playwright.sync_api import sync_playwright  # noqa: PLC0415
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            return page.content()
        finally:
            browser.close()


# ------------------------------------------------------------------ extraction
def extract_text(html: str) -> str:
    try:
        import trafilatura  # noqa: PLC0415
        text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
        if len(text) >= MIN_USEFUL_CHARS:
            return text
    except ImportError:
        pass
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return re.sub(r"\n{3,}", "\n\n", main.get_text("\n", strip=True))


def get_page_text(url: str) -> str:
    text = extract_text(fetch_static(url))
    if len(text) < MIN_USEFUL_CHARS:
        log.info("static content too thin (%d chars) -> trying headless render", len(text))
        try:
            text = extract_text(fetch_rendered(url))
        except Exception as exc:  # noqa: BLE001  (playwright missing / blocked)
            log.warning("rendered fetch failed: %s", exc)
    if len(text) < 50:
        raise RuntimeError("Could not extract meaningful content (login wall, bot block, or empty page).")
    return text


# ------------------------------------------------------------------ chunking
def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries; hard-split any single paragraph larger than `size`."""
    chunks, cur = [], ""
    for para in re.split(r"\n\s*\n|\n", text):
        while len(para) > size:                      # pathological long line
            cut = para.rfind(". ", 0, size) + 1 or size
            chunks.append(para[:cut]); para = para[cut:].lstrip()
        if len(cur) + len(para) + 1 > size and cur:
            chunks.append(cur); cur = ""
        cur += para + "\n"
    if cur.strip():
        chunks.append(cur)
    if len(chunks) > MAX_CHUNKS:                     # sample evenly instead of failing
        step = len(chunks) / MAX_CHUNKS
        chunks = [chunks[int(i * step)] for i in range(MAX_CHUNKS)]
    return chunks


# ------------------------------------------------------------------ guardrail
def word_count(s: str) -> int:
    return len(s.split())


def truncate_to_words(s: str, max_words: int) -> str:
    """Last-resort deterministic cut, preferring a sentence boundary."""
    words = s.split()
    if len(words) <= max_words:
        return s
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > len(cut) * 0.5 else cut.rstrip(",;:") + "..."


def enforce_budget(summary: str, max_words: int, llm: LLM, retries: int = 2) -> str:
    """GUARDRAIL: never return more than max_words, never return empty."""
    for attempt in range(retries + 1):
        if summary.strip() and word_count(summary) <= max_words:
            return summary.strip()
        log.warning("summary has %d words (limit %d); condensing (attempt %d)",
                    word_count(summary), max_words, attempt + 1)
        if attempt < retries:
            summary = llm(f"Condense to at most {max_words} words (target {int(max_words * .8)}). "
                          f"Keep only the key points; no preamble.\n\n{summary}")
    return truncate_to_words(summary, max_words) or "(no summary produced)"


# ------------------------------------------------------------------ summarization
def summarize_text(text: str, llm: LLM, max_words: int = 120) -> str:
    chunks = chunk_text(text)
    log.info("summarizing %d chars in %d chunk(s)", len(text), len(chunks))
    if len(chunks) == 1:
        partial = llm(f"Summarize in at most {max_words} words:\n\n{chunks[0]}")
    else:                                            # MAP: summarize each chunk
        notes = [llm(f"Summarize in 3-5 bullet points, facts only:\n\n{c}") for c in chunks]
        # REDUCE (hierarchical if the notes themselves are too long)
        while len("\n".join(notes)) > CHUNK_CHARS and len(notes) > 1:
            groups = chunk_text("\n".join(notes))
            notes = [llm(f"Merge into 5 bullet points:\n\n{g}") for g in groups]
        partial = llm(f"Write one coherent summary in at most {max_words} words from these notes:\n\n"
                      + "\n".join(notes))
    return enforce_budget(partial, max_words, llm)


def openai_llm(model: str | None = None) -> LLM:
    from openai import OpenAI  # noqa: PLC0415
    client, model = OpenAI(timeout=30, max_retries=3), model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def call(prompt: str) -> str:
        t0 = time.perf_counter()
        r = client.chat.completions.create(model=model, temperature=0.2,
                                           messages=[{"role": "user", "content": prompt}])
        log.info("llm call %.1fs, %s tokens", time.perf_counter() - t0, r.usage.total_tokens if r.usage else "?")
        return r.choices[0].message.content or ""
    return call


def summarize_url(url: str, max_words: int = 120, llm: LLM | None = None) -> str:
    return summarize_text(get_page_text(url), llm or openai_llm(), max_words)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--max-words", type=int, default=120)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(summarize_url(args.url, args.max_words))
