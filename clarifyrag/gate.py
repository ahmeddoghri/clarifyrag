"""The clarification gate: decide whether a query is answerable or ambiguous.

Two complementary signals:

* **Retrieval disagreement** — run the search and look at how retrieved evidence
  splits across *topics*. If one sense dominates, answer. If the top two senses
  have comparable mass (low margin / high entropy), the query is ambiguous.
* **Lexical under-specification** — the query names a known polyseme
  (apple, jaguar, python, mercury...) without any disambiguating term.

The gate takes the stronger of the two, so it fires even when the corpus is thin.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .corpus import Retriever, tokenize

# heads with multiple common senses, and words that resolve them
POLYSEMES = {"apple", "jaguar", "python", "mercury", "amazon"}
DISAMBIGUATORS = {
    "company", "stock", "corporation", "inc", "fruit", "grocery", "tree",
    "car", "brand", "vehicle", "animal", "cat", "wild", "language",
    "programming", "code", "snake", "reptile", "planet", "orbit", "sun",
    "element", "metal", "chemical", "hg",
}


@dataclass
class GateDecision:
    ambiguous: bool
    score: float                       # 0..1 ambiguity
    topics: list[tuple[str, float]] = field(default_factory=list)  # (topic, mass)
    reason: str = ""


def assess(query: str, retriever: Retriever, threshold: float = 0.5,
           k: int = 5) -> GateDecision:
    hits = retriever.search(query, k=k)

    # --- retrieval-disagreement signal ---
    # Ambiguity = how much mass the runner-up *sense* carries relative to the
    # winner. A well-specified query leaves the second sense negligible; a truly
    # ambiguous one has two comparable senses contending for the answer.
    topic_mass: dict[str, float] = defaultdict(float)
    for h in hits:
        topic_mass[h.doc.topic] += h.score
    ranked = sorted(topic_mass.items(), key=lambda kv: kv[1], reverse=True)

    retr_signal = 0.0
    if len(ranked) >= 2 and ranked[0][1] > 0:
        retr_signal = ranked[1][1] / ranked[0][1]       # top2 / top1, in 0..1

    # --- lexical under-specification signal ---
    toks = set(tokenize(query))
    has_poly = bool(toks & POLYSEMES)
    has_disambig = bool(toks & DISAMBIGUATORS)
    lex_signal = 0.7 if (has_poly and not has_disambig) else 0.0

    score = max(retr_signal, lex_signal)
    if not hits:
        score = max(score, 0.6)        # nothing retrieved -> better to ask
    reasons = []
    if retr_signal >= threshold:
        reasons.append(f"top senses contend ({ranked[0][0]} vs {ranked[1][0]})"
                       if len(ranked) >= 2 else "retrieval spread")
    if lex_signal >= threshold:
        reasons.append("polyseme without a disambiguating term")
    return GateDecision(
        ambiguous=score >= threshold,
        score=round(score, 4),
        topics=[(t, round(m, 4)) for t, m in ranked],
        reason="; ".join(reasons) or "single dominant sense",
    )
