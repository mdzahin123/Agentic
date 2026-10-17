"""ORIGINAL (hypothetical) script - kept for reference to show the bottlenecks.

Problems, in order of severity:
 1. Whole raw HTML (nav, ads, <script>, CSS) is sent to the LLM -> wasted tokens, noisy summary.
 2. No chunking: long pages exceed the context window -> API error or silent truncation.
 3. requests.get() has no timeout, retries, headers or size cap -> hangs / 403 / memory blowup.
 4. JS-rendered pages return an empty shell; the script never notices.
 5. No guardrail on output length: "summarize" is only a polite request.
"""
import requests
from openai import OpenAI

client = OpenAI()

def summarize(url):
    html = requests.get(url).text                      # (3) (4)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize this page:\n{html}"}],  # (1) (2) (5)
    )
    return resp.choices[0].message.content
