# 🔎 ClarifyRAG

**A clarification-aware search agent that asks before it answers.**

![CI](https://github.com/ahmeddoghri/clarifyrag/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Ask a clarifying question only when it actually helps.** In the
> benchmark, clarify-aware routing hits **100% accuracy while asking 38%
> fewer questions** than always-asking, the best utility of any policy.
> `python -m clarifyrag.eval`.

You know that coworker who answers your Slack question confidently, at
length, about the wrong thing entirely? Most RAG agents are that coworker.
Ask "what's the price of apple?" and you'll get a confident answer about
either the fruit or the company, decided by a coin flip you didn't know
was happening.

ClarifyRAG asks first when it's actually unsure. It gates on **retrieval
disagreement**: if the evidence splits across two comparable senses, it
asks. If one sense clearly dominates, it just answers. No committee
meeting required, no asking about things that are already obvious.

Runs with **zero dependencies and zero API keys** (an interpretable TF-IDF
retriever plus a heuristic gate). Point it at your own retriever and LLM
for production.

---

## The decision, live

```python
from clarifyrag import ClarifyRAG
agent = ClarifyRAG()

agent.step("tell me about jaguar")
# ASK  (ambiguity 1.00): "That could mean a few things: did you mean the
#        car, or animal sense?"   why: top senses contend (car vs animal)

agent.step("jaguar animal wild cat habitat")
# ANSWER (ambiguity 0.08): "The jaguar is a large wild cat native to the Americas."
```

It resolves multi-turn too: ask, take the hint, then answer.

```python
agent.resolve("how fast is python", intended=python_snake_doc)
# Resolution(doc=python_snake, asks=1, correct=True)
```

## Why gate on retrieval disagreement?

The ambiguity score is the mass ratio of the runner-up sense to the top
sense:

```
ambiguity = mass(2nd sense) / mass(1st sense)          # 0 = clear, 1 = a coin flip
```

Plus a lexical backstop: a known polyseme (apple, jaguar, python, mercury)
used with no disambiguating term anywhere nearby. The agent asks when
either signal crosses threshold, and you can see exactly why it asked.
No model, no vibes, just numbers you can print.

## Benchmark: knowing *when* to ask

A labeled set of ambiguous and well-specified queries. We score the
gate's decision, then compare three policies where asking costs a turn
(`utility = accuracy - 0.15 * avg_questions`):

```bash
python -m clarifyrag.eval
```
```
gate: precision=1.00  recall=1.00  F1=1.00

policy            accuracy  avg_asks   utility
always_search         0.77      0.00     0.769   fast, but wrong on the ambiguous ones
always_ask            1.00      1.00     0.850   correct, but nags on everything
clarify_aware         1.00      0.62     0.908   asks only when it pays off
```

Answering everything gets things wrong. Asking about everything gets
annoying fast. Asking selectively wins on both counts, which is a nice
change of pace for a tradeoff.

## Install

```bash
git clone https://github.com/ahmeddoghri/clarifyrag
cd clarifyrag && pip install -e .
python examples/quickstart.py
```

Or with Docker:

```bash
docker build -t clarifyrag .
docker run --rm clarifyrag
```

## Bring your own retriever

`Retriever` is a plain TF-IDF search over an in-memory corpus. Anything
that returns scored, topic-tagged hits drops in cleanly. Wire the gate to
your vector DB and the ask/answer logic doesn't change at all.

## Tests

```bash
pip install pytest && pytest -q      # 7 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure. Each one reproduces its headline number locally with no API keys:

[agentmem](https://github.com/ahmeddoghri/agentmem) · [rubricagent](https://github.com/ahmeddoghri/rubricagent) · [churnfm](https://github.com/ahmeddoghri/churnfm) · [citebench](https://github.com/ahmeddoghri/citebench) · [guardrail-gate](https://github.com/ahmeddoghri/guardrail-gate) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [vllm-cost-router](https://github.com/ahmeddoghri/vllm-cost-router) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri
