# 🔎 ClarifyRAG

**A clarification-aware search agent that asks before it answers.**

![CI](https://github.com/ahmeddoghri/clarifyrag/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Ask a clarifying question only when it actually helps.** In the benchmark,
> clarify-aware routing hits **100% accuracy while asking 38% fewer questions**
> than always-asking — highest utility of any policy. `python -m clarifyrag.eval`.

Most RAG agents answer *every* query, even the ambiguous ones — so "what's the
price of **apple**?" confidently returns the wrong thing half the time. A good
agent knows when it doesn't have enough to go on and **asks one sharp
clarifying question** instead. ClarifyRAG makes that decision principled: it
gates on **retrieval disagreement** — if the retrieved evidence splits across two
comparable *senses*, ask; if one sense dominates, answer.

Runs with **zero dependencies and zero API keys** (interpretable TF-IDF retriever
+ a heuristic gate). Point it at your own retriever/LLM for production.

> **Inspired by the mid-2026 agent literature:**
> - *When Search Agents Should Ask: DiscoBench for Clarification-Aware Deep Search* (2026).

---

## The decision, live

```python
from clarifyrag import ClarifyRAG
agent = ClarifyRAG()

agent.step("tell me about jaguar")
# ASK  (ambiguity 1.00): "That could mean a few things — did you mean the
#        animal, or car sense?"   why: top senses contend (animal vs car)

agent.step("jaguar animal wild cat habitat")
# ANSWER (ambiguity 0.08): "The jaguar is a large wild cat native to the Americas."
```

It resolves multi-turn, too — ask, take the user's hint, then answer:

```python
agent.resolve("how fast is python", intended=python_snake_doc)
# Resolution(doc=python_snake, asks=1, correct=True)
```

## Why gate on retrieval disagreement?

The ambiguity score is the **mass ratio of the runner-up sense to the top sense**:

```
ambiguity = mass(2nd sense) / mass(1st sense)          # 0 = clear, 1 = a coin-flip
```

plus a lexical backstop (a known polyseme — *apple, jaguar, python, mercury* —
used without any disambiguating term). The agent asks when either signal crosses
threshold. No model required to see *why* it asked.

## Benchmark: knowing *when* to ask (DiscoBench-style)

A labeled set of ambiguous vs. well-specified queries. We score the gate's
decision, then compare three policies where asking costs a turn
(`utility = accuracy − 0.15 × avg_questions`):

```bash
python -m clarifyrag.eval
```
```
gate: precision=1.00  recall=1.00  F1=1.00

policy            accuracy  avg_asks   utility
always_search         0.77      0.00     0.769   ← fast, but wrong on ambiguous
always_ask            1.00      1.00     0.850   ← correct, but nags on clear queries
clarify_aware         1.00      0.62     0.908   ← asks only when it pays off ✅
```

Answering everything is inaccurate; asking about everything is annoying.
Clarifying *selectively* dominates both.

## Install

```bash
git clone https://github.com/ahmeddoghri/clarifyrag
cd clarifyrag && pip install -e .
python examples/quickstart.py
```

## Bring your own retriever

`Retriever` is a plain TF-IDF over an in-memory corpus. Anything returning
scored, topic-tagged hits drops in — wire the gate to your vector DB and the
ask/answer logic is unchanged.

## Tests

```

Or with Docker:

```bash
docker build -t clarifyrag .
docker run --rm clarifyrag
```bash
pip install pytest && pytest -q      # 7 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure — each reproduces its headline number locally with no API keys:

[agentmem](https://github.com/ahmeddoghri/agentmem) · [rubricagent](https://github.com/ahmeddoghri/rubricagent) · [churnfm](https://github.com/ahmeddoghri/churnfm) · [citebench](https://github.com/ahmeddoghri/citebench) · [guardrail-gate](https://github.com/ahmeddoghri/guardrail-gate) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [vllm-cost-router](https://github.com/ahmeddoghri/vllm-cost-router) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri
