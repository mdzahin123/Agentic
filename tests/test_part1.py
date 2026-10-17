import json, sys
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "part1_design"))
from support_pipeline import (Chunk, ContactHistory, Email, SupportPipeline, load_kb,
                              risk_gate, unsupported_facts)

KB = load_kb(Path(__file__).parents[1] / "part1_design" / "kb")

def llm_factory(draft="ok", category="Billing", conf=0.9, critical=False):
    def llm(p):
        if p.startswith("Classify"): return json.dumps({"category": category, "confidence": conf})
        if p.startswith("Answer JSON"): return json.dumps({"critical": critical})
        return draft
    return llm

def test_keywords_escalate():
    for body in ["our server is down", "someone hacked my account", "all my data was deleted"]:
        assert risk_gate(Email("a@b.c", "x", body), 1)

def test_benign_not_escalated():
    assert risk_gate(Email("a@b.c", "Invoice", "Where can I download my invoice?"), 1) == []

def test_repeat_contact_more_than_3_in_7_days():
    h, t0 = ContactHistory(), datetime(2026, 10, 1)
    counts = [h.record("A@x.com", t0 + timedelta(days=i)) for i in range(4)]
    assert counts == [1, 2, 3, 4]                       # 4th contact -> >3 -> escalate
    assert h.record("a@x.com", t0 + timedelta(days=20)) == 1   # window rolled off

def test_gate_runs_before_drafting():
    calls = []
    base = llm_factory()
    def spy(p): calls.append(p[:8]); return base(p)
    r = SupportPipeline(spy, KB).process(Email("a@b.c", "Help", "data was lost!"))
    assert r.action == "human_escalation" and not any(c.startswith("Draft") for c in calls)

def test_gate_fails_safe_on_llm_error():
    def bad(p): raise RuntimeError("boom")
    assert any(r.startswith("gate_error") for r in risk_gate(Email("a@b.c", "hi", "hello"), 1, bad))

def test_refund_hallucination_blocked():
    ctx = [Chunk("R2", "Annual plans can be refunded within 30 days of purchase.")]
    assert unsupported_facts("You have 60 days to get a refund [R2]", ctx)
    assert not unsupported_facts("Annual plans can be refunded within 30 days [R2]", ctx)
    assert unsupported_facts("Refunds in 5 days [R9]", ctx)

def test_hallucinated_draft_goes_to_human():
    p = SupportPipeline(llm_factory(draft="Refunds are valid for 90 days [R2]."), KB)
    r = p.process(Email("z@z.z", "Refund", "Can I get a refund on my annual plan?"))
    assert r.action == "human_review"

def test_low_confidence_goes_to_human():
    r = SupportPipeline(llm_factory(conf=0.3), KB).process(Email("q@q.q", "hmm", "refund annual plan"))
    assert r.action == "human_review"

def test_pdf_knowledge_base(tmp_path):
    from reportlab.pdfgen import canvas
    pdf = tmp_path / "policy.pdf"
    c = canvas.Canvas(str(pdf)); c.drawString(72, 750, "Annual plans can be refunded within 30 days of purchase.")
    c.drawString(72, 730, "Domain registration is non-refundable."); c.save()
    kb = load_kb(tmp_path)
    assert kb and "30 days" in kb[0].text and kb[0].id.startswith("policy-p1")
