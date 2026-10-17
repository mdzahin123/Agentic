"""Customer-support email triage pipeline (reference implementation of Part 1).

Flow:  ingest -> RISK GATE (deterministic, pre-LLM) -> classify -> retrieve
       -> draft (grounded) -> VERIFY (refund/number guard) -> draft_ready or human_review

Every step emits a structured log line carrying a trace_id for observability.
The LLM is injected (`llm(prompt) -> str`) so the pipeline is testable offline.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable

log = logging.getLogger("support")
LLM = Callable[[str], str]


class Category(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical"
    FEEDBACK = "Feedback"
    OTHER = "Other"


@dataclass
class Email:
    sender: str
    subject: str
    body: str
    received_at: datetime = field(default_factory=datetime.now)


@dataclass
class Result:
    action: str                      # "human_escalation" | "draft_ready" | "human_review"
    reasons: list[str]
    category: Category | None = None
    draft: str | None = None
    sources: list[str] = field(default_factory=list)
    trace_id: str = ""


# ---------------------------------------------------------------- risk gate
# Deterministic on purpose: critical routing must never depend on LLM mood.
RISK_PATTERNS = {
    "data_loss": r"\b(data (is |was |has been )?(lost|gone|deleted|missing|corrupt\w*)|lost (all )?(my |our )?(data|files|backups?)|deleted (all )?(my |our )?(data|files)|files? (are|is) (gone|missing))\b",
    "outage": r"\b(outage|(site|service|server|website|app|api)s? (is |are )?(down|unreachable|not responding)|can'?t access (the )?(site|service|server)|everything is down)\b",
    "security_breach": r"\b(breach\w*|hack(ed|ers?)?|compromised|unauthori[sz]ed (access|login|transaction)|data leak\w*|leaked|phishing|stolen (credentials|password))\b",
}
CONTACT_LIMIT, CONTACT_WINDOW = 3, timedelta(days=7)


class ContactHistory:
    """Swap for Redis/SQL in production; interface stays the same."""

    def __init__(self) -> None:
        self._seen: dict[str, list[datetime]] = {}

    def record(self, sender: str, when: datetime) -> int:
        """Record contact; return number of contacts (incl. this one) in the last 7 days."""
        key = sender.strip().lower()
        stamps = [t for t in self._seen.get(key, []) if when - t <= CONTACT_WINDOW]
        stamps.append(when)
        self._seen[key] = stamps
        return len(stamps)


def risk_gate(email: Email, contacts_in_window: int, llm: LLM | None = None) -> list[str]:
    """Return list of escalation reasons (empty == safe to automate)."""
    text = f"{email.subject}\n{email.body}".lower()
    reasons = [f"keyword:{name}" for name, pat in RISK_PATTERNS.items() if re.search(pat, text)]
    if contacts_in_window > CONTACT_LIMIT:
        reasons.append(f"repeat_contact:{contacts_in_window}_in_7_days")
    # Optional semantic second opinion catches paraphrases regexes miss. It may only ADD
    # reasons; a failure or malformed answer fails SAFE (escalate), never open.
    if llm and not reasons:
        try:
            verdict = json.loads(llm(
                'Answer JSON {"critical": true|false}. Does this email report data loss, a service outage '
                f"or a security breach?\n---\n{text[:2000]}"))
            if verdict.get("critical") is True:
                reasons.append("llm:semantic_critical")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"gate_error:{type(exc).__name__}")
    return reasons


# ---------------------------------------------------------------- knowledge base
@dataclass
class Chunk:
    id: str
    text: str


def _pdf_chunks(path: Path, max_chars: int = 500) -> list[Chunk]:
    """Extract PDF text and split into ~500-char chunks with ids like `refund-p1-2`."""
    from pypdf import PdfReader  # noqa: PLC0415
    chunks: list[Chunk] = []
    for pno, page in enumerate(PdfReader(str(path)).pages, 1):
        text = re.sub(r"\s+", " ", page.extract_text() or "")
        sentences, cur, n = re.split(r"(?<=[.!?])\s+", text), "", 0
        for sent in sentences + [""]:
            if cur and (len(cur) + len(sent) > max_chars or not sent):
                n += 1
                chunks.append(Chunk(f"{path.stem}-p{pno}-{n}".replace(" ", "_"), cur.strip()))
                cur = ""
            cur += " " + sent
    return chunks


def load_kb(folder: Path) -> list[Chunk]:
    """Load .md files (tagged lines like `[R1] ...`) and .pdf files (auto-chunked)."""
    chunks: list[Chunk] = []
    for f in sorted(folder.glob("*.md")):
        for line in f.read_text().splitlines():
            m = re.match(r"\[(\w+)\]\s+(.*)", line)
            if m:
                chunks.append(Chunk(m.group(1), m.group(2)))
    for f in sorted(folder.glob("*.pdf")):
        chunks.extend(_pdf_chunks(f))
    return chunks


STOP = {"the", "and", "can", "for", "you", "your", "are", "was", "with", "that", "this", "have", "how",
        "long", "until", "from", "get", "our", "any", "but", "not", "all", "its", "may", "will"}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _tokens(s: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}


def retrieve(query: str, kb: list[Chunk], k: int = 3, min_overlap: int = 2) -> list[Chunk]:
    """Lightweight lexical retrieval (stemmed token overlap); replace with embeddings + reranker at scale."""
    q = _tokens(query)
    scored = sorted(((len(q & _tokens(c.text)), c) for c in kb), key=lambda x: -x[0])
    return [c for s, c in scored[:k] if s >= min_overlap]


# ---------------------------------------------------------------- hallucination guard
FACT_RE = re.compile(r"\$?\d+(?:[.,]\d+)?(?:\s?(?:%|days?|business days?|hours?|gb|usd))?", re.I)


def unsupported_facts(draft: str, context: list[Chunk]) -> list[str]:
    """Every number/duration in the draft must literally appear in the retrieved policy text.

    Also checks citations: the draft may cite only chunk ids that were retrieved.
    """
    ctx = " ".join(c.text for c in context).lower()
    cites = set(re.findall(r"\[(\w+)\]", draft))
    body = re.sub(r"\[\w+\]", "", draft)          # citation ids are not facts
    bad = [m.group().strip() for m in FACT_RE.finditer(body) if m.group().strip().lower() not in ctx]
    bad += [f"citation:{c}" for c in cites - {c.id for c in context}]
    return bad


# ---------------------------------------------------------------- pipeline
class SupportPipeline:
    def __init__(self, llm: LLM, kb: list[Chunk], history: ContactHistory | None = None):
        self.llm, self.kb, self.history = llm, kb, history or ContactHistory()

    def process(self, email: Email) -> Result:
        trace = uuid.uuid4().hex[:8]

        def done(res: Result) -> Result:
            res.trace_id = trace
            log.info(json.dumps({"trace": trace, "action": res.action, "reasons": res.reasons,
                                 "category": res.category.value if res.category else None}))
            return res

        # 1. FLAG BEFORE DRAFTING
        n = self.history.record(email.sender, email.received_at)
        reasons = risk_gate(email, n, self.llm)
        if reasons:
            return done(Result("human_escalation", reasons))

        # 2. classify (validated enum; unknown/low confidence -> human)
        try:
            out = json.loads(self.llm(
                'Classify into Billing|Technical|Feedback|Other. JSON {"category":..., "confidence":0-1}.\n'
                f"Subject: {email.subject}\n{email.body[:2000]}"))
            category, conf = Category(out["category"]), float(out["confidence"])
        except Exception as exc:  # noqa: BLE001
            return done(Result("human_review", [f"classify_error:{type(exc).__name__}"]))
        if conf < 0.6:
            return done(Result("human_review", [f"low_confidence:{conf}"], category))

        # 3. retrieve; no evidence -> don't improvise
        ctx = retrieve(f"{email.subject} {email.body}", self.kb)
        if not ctx:
            return done(Result("human_review", ["no_kb_evidence"], category))

        # 4. grounded draft
        ctx_txt = "\n".join(f"[{c.id}] {c.text}" for c in ctx)
        draft = self.llm(
            "Draft a polite reply using ONLY the facts below. Cite chunk ids like [R1]. "
            "If the facts do not answer the question, say a specialist will follow up. "
            "Never state any number, deadline or amount not present in the facts.\n"
            f"FACTS:\n{ctx_txt}\n\nEMAIL:\n{email.body[:2000]}")

        # 5. verify before anything is sent; one repair attempt, then human
        bad = unsupported_facts(draft, ctx)
        if bad:
            draft = self.llm(f"Rewrite removing unsupported claims {bad}. Use only:\n{ctx_txt}\n\nDRAFT:\n{draft}")
            bad = unsupported_facts(draft, ctx)
        if bad:
            return done(Result("human_review", [f"unsupported_claims:{bad}"], category, draft))
        return done(Result("draft_ready", [], category, draft, [c.id for c in ctx]))
