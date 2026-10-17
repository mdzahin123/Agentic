"""Live test of all three parts against the real OpenAI API.

    python check_setup.py

Each test prints: what it checks, what we gave it, what we expected, what we got, PASS/FAIL.
Costs a few cents at most.
"""
import logging
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

os.system("")                       # lets Windows terminals show colours
logging.disable(logging.CRITICAL)   # keep the screen clean: no technical log lines

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
for sub in ("part1_design", "part2_scraper", "part3_agent"):
    sys.path.insert(0, str(ROOT / sub))

GREEN, RED, BOLD, DIM, RESET = "\033[92m", "\033[91m", "\033[1m", "\033[2m", "\033[0m"
WIDTH = 78
summary: list[tuple[str, bool]] = []


def wrap(label: str, text: str) -> str:
    """Aligned 'Label : text' line that wraps neatly."""
    head = f"   {label:<12}: "
    paras = str(text).split("\n")
    out = [textwrap.fill(" ".join(paras[0].split()), WIDTH, initial_indent=head,
                         subsequent_indent=" " * len(head))]
    out += [textwrap.fill(" ".join(p.split()), WIDTH, initial_indent=" " * len(head),
                          subsequent_indent=" " * len(head)) for p in paras[1:] if p.strip()]
    return "\n".join(out)


def part_header(title: str) -> None:
    print(f"\n{BOLD}{'=' * WIDTH}\n {title}\n{'=' * WIDTH}{RESET}")


def run_test(name: str, gave: str, expected: str, fn) -> None:
    """fn() -> (ok: bool, what_happened: str). Errors become a readable FAIL."""
    try:
        ok, got = fn()
    except Exception as exc:  # noqa: BLE001
        ok, got = False, f"Crashed: {type(exc).__name__}: {str(exc)[:200]}"
    summary.append((name, ok))
    status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"\n {BOLD}{name}{RESET}")
    print(wrap("We gave it", gave))
    print(wrap("Expected", expected))
    print(wrap("It did", got))
    print(f"   {'Result':<12}: {status}")


# ---------------------------------------------------------------- start
print(f"{BOLD}\n LIVE TEST: checking that all 3 parts work with the real AI{RESET}")
key = os.getenv("OPENAI_API_KEY", "")
if not key.startswith("sk-"):
    sys.exit(f"\n{RED}Your OpenAI key is missing.{RESET} Open the .env file and put it after OPENAI_API_KEY=")

from openai import OpenAI                                       # noqa: E402
from openai_llm import make_openai_llm                          # noqa: E402
from support_pipeline import Email, SupportPipeline, load_kb    # noqa: E402
import fixed_summarizer as fs                                   # noqa: E402
from agent import DocAgent                                      # noqa: E402

# ---------------------------------------------------------------- Part 1
part_header("PART 1 - Support email sorter  (sends risky emails to humans, drafts safe replies)")
pipe = SupportPipeline(make_openai_llm(), load_kb(ROOT / "part1_design" / "kb"))
REASONS = {"keyword:data_loss": "mentions lost data", "keyword:outage": "mentions an outage",
           "keyword:security_breach": "mentions a security breach",
           "llm:semantic_critical": "the AI judged it critical"}


def t1():
    r = pipe.process(Email("a@x.com", "Help", "All my files have vanished after the update!"))
    why = ", ".join(REASONS.get(x, x) for x in r.reasons)
    api_error = any(x.startswith("gate_error") for x in r.reasons)   # an AI failure must not count as a pass
    return (r.action == "human_escalation" and r.draft is None and not api_error,
            f"{'Sent to a HUMAN' if r.action == 'human_escalation' else 'Did NOT escalate'} ({why}); "
            f"reply written: {'yes' if r.draft else 'no'}")


def t2():
    r = pipe.process(Email("b@x.com", "Refund", "Can I get a refund on my annual plan and how long does it take?"))
    ok = r.action in ("draft_ready", "human_review") and not any(x.startswith(("gate_error", "classify_error")) for x in r.reasons)
    msg = "Wrote a reply that passed the fact-checker" if r.action == "draft_ready" else \
          f"Sent to a human for safety ({r.reasons})"
    return ok, msg + (f'\nReply: "{" ".join(r.draft.split())[:300]}"' if r.draft else "")


def t3():
    for _ in range(3):
        pipe.process(Email("c@x.com", "Hi", "Where is my invoice?"))
    r = pipe.process(Email("c@x.com", "Hi", "Where is my invoice?"))
    return r.action == "human_escalation", f"4th email: {r.action.replace('_', ' ')} ({', '.join(r.reasons)})"


run_test("1. Dangerous email goes to a human", "email: 'All my files have vanished after the update!'",
         "sent to a human and NO automatic reply", t1)
run_test("2. Refund question gets a checked reply", "email: 'Can I get a refund on my annual plan?'",
         "reply that only uses real policy facts (30 days)", t2)
run_test("3. Customer who keeps emailing goes to a human", "the same customer emails 4 times",
         "the 4th email is sent to a human", t3)

# ---------------------------------------------------------------- Part 2
part_header("PART 2 - Summarizer  (shortens long text, never exceeds the word limit)")
LONG = "\n\n".join(f"Section {i}. Nimbus Cloud launched plan number {i} with new features and pricing changes."
                   for i in range(400))


def t4():
    s = fs.summarize_text(LONG, fs.openai_llm(), max_words=60)
    n = fs.word_count(s)
    return 0 < n <= 60, f"Wrote a summary of {n} words (limit 60).\nSummary: \"{' '.join(s.split())[:300]}\""


run_test("4. Very long text becomes a short summary", f"a text of about {len(LONG.split())} words, limit 60 words",
         "summary of 60 words or fewer", t4)

# ---------------------------------------------------------------- Part 3
part_header("PART 3 - Chat helper  (answers from a document, remembers you, uses a calculator)")
agent = DocAgent(OpenAI(timeout=30, max_retries=3), (ROOT / "part3_agent" / "sample_doc.md").read_text(),
                 os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
try:
    agent.ask("Hi, I'm Danish")
except Exception as exc:  # noqa: BLE001
    print(f"{RED}Could not talk to OpenAI: {str(exc)[:200]}{RESET}")


def t5():
    a = agent.ask("What's my name?")
    return "danish" in a.lower(), f'It answered: "{a}"'


def t6():
    before = agent.tool_calls_made
    a = agent.ask("What is the refund window for annual plans?")
    used = agent.tool_calls_made - before
    return used == 0 and "30" in a, f'Answered "{a}" and used the calculator {used} time(s)'


def t7():
    before = agent.tool_calls_made
    a = agent.ask("How much is the Pro plan for a full year with the annual discount?")
    used = agent.tool_calls_made - before
    return used >= 1 and "144" in a, f'Used the calculator {used} time(s) and answered: "{a}"'


run_test("5. It remembers your name", "you said 'Hi, I'm Danish', then asked 'What's my name?'",
         "it says Danish", t5)
run_test("6. No calculator for a simple question", "question: 'What is the refund window for annual plans?'",
         "answers 30 days, calculator used 0 times", t6)
run_test("7. Calculator for real maths", "question: 'Pro plan for a full year with the annual discount?' ($15 x 12 x 0.8)",
         "uses the calculator, answers $144", t7)

# ---------------------------------------------------------------- summary
passed = sum(ok for _, ok in summary)
print(f"\n{BOLD}{'=' * WIDTH}\n SUMMARY\n{'=' * WIDTH}{RESET}")
for name, ok in summary:
    print(f"  {GREEN + 'PASS' if ok else RED + 'FAIL'}{RESET}  {name}")
colour = GREEN if passed == len(summary) else RED
print(f"\n {colour}{BOLD}{passed} of {len(summary)} tests passed.{RESET}"
      + (" Everything works." if passed == len(summary) else " Copy the FAIL lines above and send them to me."))
sys.exit(0 if passed == len(summary) else 1)
