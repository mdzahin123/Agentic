"""Document Q&A agent with conversation memory and an on-demand calculator tool.

Agentic pattern: a ReAct-style loop built on raw OpenAI tool-calling (no framework).
    messages -> model -> (tool_calls? -> execute -> append results -> model ...) -> final answer
The MODEL decides whether to call `calculator`; we only supply the tool and safety rails
(max iterations, argument validation, error-as-observation, trace logging).

Run:  python agent.py            (interactive; needs OPENAI_API_KEY)
"""
from __future__ import annotations

import ast
import json
import logging
import operator as op
import os
from pathlib import Path

log = logging.getLogger("agent")

# ------------------------------------------------------------------ tool
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic via the AST. Never uses eval(): no names, calls or attributes."""
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    if len(expr) > 200:
        raise ValueError("expression too long")
    return ev(ast.parse(expr.strip(), mode="eval").body)


def calculator(expression: str) -> str:
    try:
        v = safe_eval(expression)
        return str(round(v, 6) if isinstance(v, float) else v)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:  # noqa: BLE001 - returned to the model as an observation
        return f"Error: {exc}"


TOOLS = [{"type": "function", "function": {
    "name": "calculator",
    "description": ("Evaluate an arithmetic expression (+ - * / ** %, parentheses). Use ONLY when the answer "
                    "requires non-trivial arithmetic (multi-step, percentages, totals). Do NOT use for "
                    "lookups, quoting document facts, or trivial math."),
    "parameters": {"type": "object", "properties": {"expression": {"type": "string",
                   "description": "e.g. '15 * 12 * (1 - 0.20)'"}}, "required": ["expression"]}}}]
TOOL_IMPL = {"calculator": calculator}

SYSTEM = """You are a helpful assistant that answers questions about the document below.
Rules:
- Answer ONLY from the document. If it is not covered, say you can't find it in the document.
- Remember what the user tells you (name, preferences, earlier questions) and use it naturally.
- Call the calculator tool only when real arithmetic is needed; otherwise answer directly.
- Be concise.

<document>
{doc}
</document>"""


# ------------------------------------------------------------------ agent
class DocAgent:
    def __init__(self, client, doc_text: str, model: str = "gpt-4o-mini",
                 max_steps: int = 5, max_history: int = 30, memory_file: Path | None = None):
        self.client, self.model, self.max_steps, self.max_history = client, model, max_steps, max_history
        self.system = {"role": "system", "content": SYSTEM.format(doc=doc_text)}
        self.memory_file = memory_file
        self.history: list[dict] = self._load()          # conversation memory (user/assistant/tool turns)
        self.tool_calls_made = 0

    # -- memory ---------------------------------------------------
    def _load(self) -> list[dict]:
        if self.memory_file and self.memory_file.exists():
            try:
                return json.loads(self.memory_file.read_text())
            except json.JSONDecodeError:
                log.warning("memory file corrupt; starting fresh")
        return []

    def _save(self) -> None:
        if self.memory_file:
            self.memory_file.write_text(json.dumps(self.history, indent=1))

    def _trim(self) -> None:
        """Keep the last N messages, never starting on an orphaned tool message."""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            while self.history and self.history[0]["role"] in ("tool", "assistant"):
                self.history.pop(0)

    # -- loop -----------------------------------------------------
    def ask(self, user_msg: str) -> str:
        self.history.append({"role": "user", "content": user_msg})
        for step in range(self.max_steps):
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[self.system, *self.history], tools=TOOLS)
            msg = resp.choices[0].message
            entry: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [{"id": t.id, "type": "function",
                                        "function": {"name": t.function.name, "arguments": t.function.arguments}}
                                       for t in msg.tool_calls]
            self.history.append(entry)
            if not msg.tool_calls:                               # model chose to answer directly
                self._trim(); self._save()
                return entry["content"]
            for t in msg.tool_calls:                             # model chose to act
                self.tool_calls_made += 1
                result = self._run_tool(t.function.name, t.function.arguments)
                log.info("step=%d tool=%s args=%s -> %s", step, t.function.name, t.function.arguments, result)
                self.history.append({"role": "tool", "tool_call_id": t.id, "content": result})
        self._save()
        return "Sorry, I couldn't finish that request within the step limit."

    @staticmethod
    def _run_tool(name: str, raw_args: str) -> str:
        fn = TOOL_IMPL.get(name)
        if not fn:
            return f"Error: unknown tool {name}"
        try:
            return fn(**json.loads(raw_args))
        except Exception as exc:  # noqa: BLE001 - observation, lets the model recover
            return f"Error: bad arguments ({exc})"


def main() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    here = Path(__file__).parent
    agent = DocAgent(OpenAI(timeout=30, max_retries=3), (here / "sample_doc.md").read_text(),
                     os.getenv("OPENAI_MODEL", "gpt-4o-mini"), memory_file=here / "memory.json")
    print("Ask about the Nimbus document ('quit' to exit). Memory persists in memory.json (delete to reset).")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            try:
                print("agent>", agent.ask(q))
            except Exception as exc:  # noqa: BLE001 - API outage etc.: stay alive
                log.error("request failed: %s", exc)
                print("agent> Sorry, something went wrong. Please try again.")


if __name__ == "__main__":
    main()
