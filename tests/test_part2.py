import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "part2_scraper"))
import fixed_summarizer as fs

def test_extract_strips_boilerplate():
    html = "<html><body><nav>MENU</nav><script>var x=1</script><article>" + "Real content here. " * 30 + "</article><footer>FOOT</footer></body></html>"
    t = fs.extract_text(html)
    assert "Real content" in t and "MENU" not in t and "var x" not in t

def test_chunking_respects_size_and_handles_giant_line():
    chunks = fs.chunk_text("word. " * 20000, size=1000)
    assert all(len(c) <= 1100 for c in chunks) and len(chunks) <= fs.MAX_CHUNKS

def test_map_reduce_on_long_doc_and_budget_enforced():
    calls = []
    def llm(p):
        calls.append(p)
        return "very long " * 500 if "at most 50 words" in p and "notes" in p else "short note"
    out = fs.summarize_text(("Paragraph of text.\n\n" * 5000), llm, max_words=50)
    assert fs.word_count(out) <= 50 and len(calls) > 3

def test_guardrail_hard_truncates_when_llm_ignores_limit():
    out = fs.enforce_budget("Sentence one. " * 200, 20, lambda p: "Sentence one. " * 200)
    assert fs.word_count(out) <= 20

def test_guardrail_never_returns_empty():
    assert fs.enforce_budget("", 20, lambda p: "") == "(no summary produced)"
