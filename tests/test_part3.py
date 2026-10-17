import json, sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).parents[1] / "part3_agent"))
from agent import DocAgent, calculator, safe_eval

def reply(content=None, tool=None):
    tc = [NS(id="c1", function=NS(name="calculator", arguments=json.dumps({"expression": tool})))] if tool else None
    return NS(choices=[NS(message=NS(content=content, tool_calls=tc))])

class FakeClient:
    """Scripted model: lets us test the loop, memory and tool plumbing deterministically."""
    def __init__(self, script): self.script, self.seen = list(script), []
        
    @property
    def chat(self): return NS(completions=self)
    def create(self, **kw): self.seen.append(kw["messages"]); return self.script.pop(0)

def test_calculator_basic_and_safety():
    assert safe_eval("15 * 12 * (1 - 0.20)") == 144
    for bad in ["__import__('os').system('ls')", "open('x')", "2**9999", "a+1"]:
        assert calculator(bad).startswith("Error")
    assert calculator("1/0") == "Error: division by zero"

def test_tool_called_only_when_model_requests_it():
    c = FakeClient([reply("Refunds are 14 days.")])
    a = DocAgent(c, "doc"); a.ask("Refund window for monthly plans?")
    assert a.tool_calls_made == 0

def test_tool_loop_and_result_fed_back():
    c = FakeClient([reply(tool="15*12*0.8"), reply("Pro annual costs $144.")])
    a = DocAgent(c, "doc"); out = a.ask("Annual price of Pro?")
    assert a.tool_calls_made == 1 and "144" in out
    assert any(m["role"] == "tool" and m["content"] == "144.0" for m in c.seen[1])

def test_memory_persists_across_turns():
    c = FakeClient([reply("Nice to meet you, Sam."), reply("Your name is Sam.")])
    a = DocAgent(c, "doc"); a.ask("Hi, I'm Sam"); a.ask("What's my name?")
    assert any(m.get("content") == "Hi, I'm Sam" for m in c.seen[1])

def test_step_limit_prevents_infinite_loop():
    c = FakeClient([reply(tool="1+1")] * 10)
    a = DocAgent(c, "doc", max_steps=3)
    assert "step limit" in a.ask("loop")

def test_memory_file_roundtrip(tmp_path):
    f = tmp_path / "m.json"
    DocAgent(FakeClient([reply("hi Sam")]), "doc", memory_file=f).ask("I'm Sam")
    assert DocAgent(FakeClient([]), "doc", memory_file=f).history[0]["content"] == "I'm Sam"
