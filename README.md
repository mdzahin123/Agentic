# Agentic Architect Challenge

Three small AI programs built for the assessment. Everything is Python and uses the OpenAI API.

| Part | Folder | What it does |
|---|---|---|
| 1 | `part1_design/` | Sorts customer support emails: risky ones go to a human, safe ones get a fact-checked draft reply |
| 2 | `part2_scraper/` | Summarizes web pages, even long or complicated ones, and never exceeds a word limit |
| 3 | `part3_agent/` | Chat agent that answers questions about one document, remembers the conversation, and uses a calculator only when needed |

---

## How to run

**Requirements:** Python 3.10 or newer and an OpenAI API key.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key: copy .env.example to .env, then edit .env
copy .env.example .env            # Windows   (Mac/Linux: cp .env.example .env)
#    OPENAI_API_KEY=sk-your-key-here
```

| Command | What it does |
|---|---|
| `python check_setup.py` | Live test of all three parts with the real AI (7 tests, a few cents) |
| `python -m pytest -q tests` | 20 offline tests, no API key needed |
| `python part1_design/run_live.py` | Runs 5 sample emails through Part 1 |
| `python part2_scraper/fixed_summarizer.py <url> --max-words 120` | Summarizes a web page |
| `python part3_agent/agent.py` | Chat with the Part 3 agent (type `quit` to exit) |

Try this in the Part 3 chat: `Hi, I'm Danish` > `What's the refund window for annual plans?` (no calculator) > `How much is the Pro plan for a year with the annual discount?` (uses calculator, answer $144) > `What's my name?` (memory).

---

## Part 1: Support email system

**Pipeline** (`part1_design/support_pipeline.py`). Every email moves through these steps in order:

| Step | What it does | If something goes wrong |
|---|---|---|
| 1. Contact history | Notes who wrote and counts their emails in the last 7 days | - |
| 2. Risk gate | Checks for data loss, service outage, security breach, or more than 3 contacts in 7 days | Sent straight to a human, no reply is drafted |
| 3. Classify | AI labels the email Billing, Technical, Feedback or Other, with a confidence score | Low confidence or a bad answer goes to human review |
| 4. Retrieve | Finds matching facts in the knowledge base (`kb/`, accepts `.md` and `.pdf`) | No matching facts goes to human review |
| 5. Draft | AI writes a reply using only those facts and cites them (like `[R2]`) | - |
| 6. Verify | Every number and citation in the reply must exist in the retrieved facts | One repair attempt, then human review |
| 7. Result | `draft_ready`, `human_review` or `human_escalation` | - |

**Key design choices**
- The risk gate runs *before* any drafting, as the brief requires. It uses fixed rules, not AI, so it is predictable and explainable. An AI second opinion can add escalations but never remove one.
- "More than 3 contacts" counts the current email, so the 4th email in 7 days escalates.
- **Refund hallucination:** the AI only sees real policy text, must cite it, and a separate non-AI verifier rejects any number or citation that isn't in that text. No supporting facts means a human takes over.
- Every failure (AI error, bad format, low confidence) falls back to a human, never to an automatic send.
- Each email gets a trace ID and one structured log line (route, reasons, category).

**Other files:** `openai_llm.py` connects the pipeline to OpenAI. `run_live.py` is a live demo. `kb/` holds the sample refund policy and FAQ.

## Part 2: Web page summarizer

`original_script.py` is the hypothetical broken script, with its problems listed in comments. `fixed_summarizer.py` is the fix.

**The bottlenecks and fixes**

| Problem in the original | Fix |
|---|---|
| Raw HTML (menus, ads, scripts) sent to the AI | Extract only the article text (trafilatura, with a BeautifulSoup fallback) |
| Long pages overflow the AI's context | Split into paragraph-aware chunks, summarize each, then combine (map-reduce) |
| Plain `requests.get`: no timeout, retries or size limit | Timeouts, retries with backoff, 5 MB cap, content-type check |
| JavaScript pages return an empty shell | If the text is too thin, fall back to a headless browser (optional Playwright) |
| Nothing enforces "keep it short" | **Guardrail:** count words, ask the AI to condense (up to 2 times), then cut at a sentence boundary so the limit always holds |

## Part 3: Document chat agent

`part3_agent/agent.py` is a loop built on raw OpenAI tool-calling (no framework). The document is `sample_doc.md`, a made-up hosting company's plans and policies.

1. Your message is added to the conversation list.
2. The whole list plus the document goes to the AI.
3. The AI either answers directly, or asks to use the calculator.
4. If it asks, the code runs the calculator, adds the result to the list, and asks the AI again.

- **Memory:** the AI forgets between calls, so the full conversation is sent every time. It is also saved to `memory.json` so it survives restarts. Very old messages are trimmed.
- **Tool use:** the *AI* decides whether to call the calculator. Its description limits it to real arithmetic.
- **Safety:** the calculator parses maths safely and never uses `eval()`. A maximum of 5 steps prevents endless loops. Tool errors are returned to the AI so it can recover, and an API failure doesn't crash the chat.

---

## Trade-offs and failure handling

| Choice | Benefit | Cost / mitigation |
|---|---|---|
| Rule-based risk gate | Fast, auditable, never skipped | Some false alarms; over-escalating is safer than missing an outage |
| Strict number-matching verifier | Hard guarantee on refund facts | Rejects valid paraphrases ("two weeks"), so they go to a human |
| Whole document in the prompt (Part 3) | Simple, no retrieval errors | Doesn't scale to big documents; use retrieval for those |
| Simple word-overlap search (Part 1) | No extra dependencies | Lower recall; production should use embeddings |
| In-memory contact history | Easy to test | Resets on restart; production should use a database |

| Failure | Handling |
|---|---|
| AI timeout or 5xx error | Retries with backoff, then falls back to a human |
| Malformed AI output | Validated, then human review |
| Prompt injection in an email | Email is treated as data; the verifier is independent of the AI |
| Heavy load | Not implemented in code; the design is stateless workers behind a queue with per-sender rate limits |
| Out-of-date knowledge base | Draft records which facts it used, so re-indexing is traceable |

**Testing and monitoring:** 20 offline tests with a mocked AI cover gate ordering, the 4th-contact rule, hallucination blocking, PDF loading, chunking, the length guardrail, tool decisions, memory and the step limit. Part 1 logs a trace per email; useful metrics would be escalation rate, verifier-rejection rate and p95 latency.

## Project layout
```
part1_design/   support_pipeline.py, openai_llm.py, run_live.py, kb/
part2_scraper/  original_script.py, fixed_summarizer.py
part3_agent/    agent.py, sample_doc.md
tests/          test_part1.py, test_part2.py, test_part3.py
check_setup.py  requirements.txt  .env.example
```
