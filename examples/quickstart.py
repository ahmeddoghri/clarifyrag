"""Watch the agent decide when to ask vs. answer.
Run: python examples/quickstart.py
"""
from clarifyrag import ClarifyRAG
from clarifyrag.corpus import DEFAULT_CORPUS

agent = ClarifyRAG()

for q in [
    "tell me about jaguar",              # ambiguous -> should ask
    "jaguar animal wild cat habitat",    # specified -> should answer
    "what is the price of apple",        # ambiguous -> should ask
    "boiling point of water at sea level"  # unambiguous fact -> answer
]:
    t = agent.step(q)
    print(f"\nQ: {q}")
    print(f"   [{t.action.upper()}  ambiguity={t.gate.score:.2f}] {t.content}")
    print(f"   why: {t.gate.reason}")

# Full multi-turn resolution against a simulated user:
snake = next(d for d in DEFAULT_CORPUS if d.id == "python_snake")
res = agent.resolve("how fast is python", intended=snake)
print(f"\nResolved 'how fast is python' -> {res.doc.id} "
      f"(correct={res.correct}, asked {res.asks}x)")
