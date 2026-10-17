"""Run the Part 1 pipeline against the REAL OpenAI API:  python part1_design/run_live.py"""
import logging
from pathlib import Path

from dotenv import load_dotenv

from openai_llm import make_openai_llm
from support_pipeline import Email, SupportPipeline, load_kb

load_dotenv(Path(__file__).parents[1] / ".env")
logging.basicConfig(level=logging.INFO, format="%(message)s")

pipe = SupportPipeline(make_openai_llm(), load_kb(Path(__file__).parent / "kb"))
emails = [
    Email("amy@x.com", "Refund", "Hi, I bought an annual plan 10 days ago. Can I get a refund and how long will it take?"),
    Email("bob@x.com", "URGENT", "Everything I uploaded has vanished from my account after your update!"),
    Email("cat@x.com", "Refund policy", "Can I get a refund on my domain registration?"),
    Email("dan@x.com", "Love it", "Just wanted to say the new dashboard is great, thanks!"),
    Email("eve@x.com", "Warranty", "What is your lifetime hardware warranty policy?"),   # not in KB -> human
]
for e in emails:
    r = pipe.process(e)
    print(f"\n[{e.sender}] {e.subject}\n  -> {r.action}  reasons={r.reasons}  category={r.category}")
    if r.draft:
        print("  draft:", r.draft)
