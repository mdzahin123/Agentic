# Agentic Architect Challenge

Three small AI programs written in Python that use the OpenAI API.

| Part | Folder | What it does |
|---|---|---|
| 1 | `part1_design/` | Sorts one customer support email: risky ones go to a human, safe ones get a fact-checked draft reply |
| 2 | `part2_scraper/` | Summarizes a web page and keeps the summary under a word limit |
| 3 | `part3_agent/` | Chat helper that answers questions about one document, remembers the conversation, and uses a calculator when the AI decides it needs one |

## How to run

Requirements: Python 3.10 or newer, and an OpenAI API key.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # Windows   (Mac/Linux: cp .env.example .env)
# then open .env and set OPENAI_API_KEY=sk-your-key-here
```

| Command | What it does |
|---|---|
| `python check_setup.py` | Runs 7 tests of all three parts against the real AI and prints PASS or FAIL for each |
| `python -m pytest -q tests` | Runs 20 tests with a pretend AI (no key needed, no cost) |
| `python part1_design/run_live.py` | Runs 5 sample emails through Part 1 with the real AI |
| `python part2_scraper/fixed_summarizer.py <web address> --max-words 120` | Summarizes a web page |
| `python part3_agent/agent.py` | Starts the chat (type `quit` to exit) |

Try in the chat: `Hi, I'm Danish`, then `What's the refund window for annual plans?` (answered directly), then `How much is the Pro plan for a year with the annual discount?` (uses the calculator, answer $144), then `What's my name?` (memory).

---

## Part 1: Support email sorter

Input: one email (sender, subject, message). Output: one of `human_escalation`, `human_review` or `draft_ready`.

1. **Counts contacts.** It remembers who wrote and when, and counts that sender's emails in the last 7 days (including the current one). This memory resets when the program stops.
2. **Safety check (before any drafting).** Fixed rules search for lost data, a service being down, or a security breach. More than 3 contacts in 7 days also triggers it. Any hit gives `human_escalation` and no reply is written. If the rules find nothing, the AI is asked as a second opinion whether the email is critical; if it says yes, or the AI call fails, the email is escalated.
3. **Labels the email.** The AI returns a label (Billing, Technical, Feedback or Other) and a confidence score. An unreadable answer, an unknown label or confidence under 60% gives `human_review`.
4. **Finds facts.** It matches the email against the knowledge base in `kb/` (`.md` and `.pdf` files) by shared words. No match gives `human_review`.
5. **Writes a reply.** The AI is told to use only those facts and to cite them (like `[R2]`).
6. **Checks the reply.** Every number in the reply must appear in the facts, and every citation must be one of the facts provided. If not, the AI gets one chance to rewrite it. If it is still wrong, the result is `human_review`. This is what stops made-up refund rules.
7. **Result.** If all checks pass: `draft_ready`, with the reply and the fact IDs used.

Each email gets a short ID and one log line (result, reasons, label).

Files: `support_pipeline.py` (the logic), `openai_llm.py` (connects to OpenAI), `run_live.py` (demo), `kb/` (sample refund policy and FAQ).

## Part 2: Web page summarizer

Input: a web address and a word limit. `original_script.py` is the hypothetical broken script, with its problems in the comments. `fixed_summarizer.py` is the fixed version.

| Problem in the original | What the fixed version does |
|---|---|
| Whole messy page given to the AI | Keeps only the article text (menus, ads, hidden code removed) |
| Long pages did not fit | Cuts the text into pieces at paragraph breaks (at most 40 pieces), summarizes each, then combines the summaries |
| Careless download | Time limits, retries on temporary website errors, stops reading after 5 MB, checks it is a web page |
| Pages that fill in content after opening came back empty | If almost no text is found, tries a hidden browser. This needs the optional Playwright add-on (see below) and has not been tested |
| Nothing enforced a short summary | **Length guardrail:** counts words; if too long, asks the AI to shorten it (up to 2 times); if still too long, the code cuts it at a sentence ending. An empty summary is never returned |

Optional add-on for the hidden browser: `pip install playwright` then `playwright install chromium`.

## Part 3: Document chat helper

Answers from `sample_doc.md` (a made-up hosting company's plans and policies).

- **Memory:** all messages are kept in a list and sent to the AI every time. The list is saved to `memory.json` so it survives restarts, and only the latest 30 messages are kept. Delete `memory.json` to start fresh.
- **Calculator:** the AI decides whether to use it, and the code runs it only when asked. It accepts basic maths (add, subtract, multiply, divide, powers, remainder) and refuses anything else.
- **Loop:** after the calculator runs, its result goes back to the AI, which writes the final answer. It stops after at most 5 rounds.
- **Errors:** a failed AI request prints a message and the chat continues. A calculator error is passed back to the AI.

## Tests

- `tests/` has 20 tests that use a pretend AI. They cover the safety check running before drafting, the 4th-email rule, blocking made-up refund numbers, PDF loading, text splitting, the summary length guardrail, calculator safety, the AI-decides tool loop, memory, and the 5-round limit.
- `check_setup.py` runs 7 tests against the real AI.

## Trade-offs and known limits

- The safety check uses fixed rules, so it is predictable but may flag emails that are not urgent.
- The fact-checker is strict: a reply that rewords a number (for example "two weeks" instead of "14 days") is rejected and goes to a human.
- Part 1 finds facts by shared words, which is simple but finds fewer matches than smarter search.
- Part 1 does not send emails; it only produces a result and a draft.
- If the OpenAI call fails while writing or rewriting the reply (steps 5 and 6), the error is not caught and that email stops with an error. Failures in the safety check and labelling steps are handled.
- The contact history is not saved when the program stops.
- Part 3 gives the whole document to the AI, which works for one short document but not for large ones.
- The hidden-browser fallback in Part 2 is untested.

## Layout
```
part1_design/   support_pipeline.py, openai_llm.py, run_live.py, kb/
part2_scraper/  original_script.py, fixed_summarizer.py
part3_agent/    agent.py, sample_doc.md
tests/          test_part1.py, test_part2.py, test_part3.py
check_setup.py  requirements.txt  .env.example
```
