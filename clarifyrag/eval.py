"""DiscoBench-style evaluation: knowing *when* to ask.

Each item is a query, the doc the user actually wanted, and whether the query is
genuinely ambiguous. We measure two things:

1. **Gate quality**: precision/recall/F1 of the "should I ask?" decision.
2. **End-task utility** under three policies, where asking costs a turn:
   * ``always_search``: never clarify (fails on ambiguous queries),
   * ``always_ask``: always clarify (wastes turns on clear queries),
   * ``clarify_aware``: ClarifyRAG's gated policy.

Utility = accuracy - ask_penalty * (avg clarifying questions). The gated policy
should win: it asks only when it pays off.

    python -m clarifyrag.eval
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from .agent import ClarifyRAG
from .corpus import Retriever


@dataclass
class Item:
    query: str
    intended_id: str
    ambiguous: bool


BENCH: list[Item] = [
    # genuinely ambiguous
    Item("what is the price of apple", "apple_inc", True),
    Item("is apple healthy to eat", "apple_fruit", True),
    Item("tell me about jaguar", "jaguar_animal", True),
    Item("how fast is python", "python_snake", True),
    Item("what is mercury", "mercury_planet", True),
    Item("how big is python", "python_snake", True),
    # clear / well-specified
    Item("apple stock company share price", "apple_inc", False),
    Item("jaguar animal wild cat habitat", "jaguar_animal", False),
    Item("python programming language syntax", "python_lang", False),
    Item("mercury planet orbit around the sun", "mercury_planet", False),
    Item("how tall is the eiffel tower", "eiffel", False),
    Item("speed of light in a vacuum", "light", False),
    Item("boiling point of water at sea level", "water", False),
]


def _f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def run(ask_penalty: float = 0.15):
    retr = Retriever()
    by_id = {d.id: d for d in retr.docs}
    agent = ClarifyRAG(retr)

    # 1. gate quality
    tp = fp = fn = 0
    for it in BENCH:
        asked = agent.step(it.query).action == "ask"
        if asked and it.ambiguous:
            tp += 1
        elif asked and not it.ambiguous:
            fp += 1
        elif not asked and it.ambiguous:
            fn += 1
    prec, rec, f1 = _f1(tp, fp, fn)

    # 2. policy utility
    def eval_policy(mode: str):
        correct = asks_total = 0
        for it in BENCH:
            intended = by_id[it.intended_id]
            if mode == "always_search":
                hits = retr.search(it.query, k=1)
                doc = hits[0].doc if hits else None
                correct += doc is intended
            elif mode == "always_ask":
                q = f"{it.query} {intended.hint}".strip()
                hits = retr.search(q, k=1)
                doc = hits[0].doc if hits else None
                correct += doc is intended
                asks_total += 1
            else:  # clarify_aware
                res = agent.resolve(it.query, intended)
                correct += res.correct
                asks_total += res.asks
        acc = correct / len(BENCH)
        avg_asks = asks_total / len(BENCH)
        return acc, avg_asks, acc - ask_penalty * avg_asks

    policies = {m: eval_policy(m) for m in
                ["always_search", "always_ask", "clarify_aware"]}
    return (prec, rec, f1), policies


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ask-penalty", type=float, default=0.15)
    args = p.parse_args()
    (prec, rec, f1), policies = run(args.ask_penalty)
    print(f"gate: precision={prec:.2f}  recall={rec:.2f}  F1={f1:.2f}\n")
    print(f"{'policy':<16}{'accuracy':>10}{'avg_asks':>10}{'utility':>10}")
    for name, (acc, asks, util) in policies.items():
        print(f"{name:<16}{acc:>10.2f}{asks:>10.2f}{util:>10.3f}")


if __name__ == "__main__":
    main()
