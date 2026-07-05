"""ClarifyRAG: a search agent that asks before it answers when it should."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .corpus import Doc, Retriever
from .gate import GateDecision, assess


@dataclass
class Turn:
    action: str            # "ask" | "answer"
    content: str           # the clarifying question, or the answer
    gate: GateDecision
    doc: Optional[Doc] = None


class ClarifyRAG:
    def __init__(self, retriever: Optional[Retriever] = None,
                 ask_threshold: float = 0.5) -> None:
        self.retriever = retriever or Retriever()
        self.ask_threshold = ask_threshold

    def step(self, query: str) -> Turn:
        """One decision: either ask a clarifying question, or answer."""
        gate = assess(query, self.retriever, self.ask_threshold)
        if gate.ambiguous:
            return Turn("ask", self._clarify(gate), gate)
        hits = self.retriever.search(query, k=1)
        if not hits:
            return Turn("ask", "I couldn't find anything — can you rephrase or "
                        "add detail?", gate)
        doc = hits[0].doc
        return Turn("answer", doc.answer, gate, doc)

    def _clarify(self, gate: GateDecision) -> str:
        senses = [t for t, _ in gate.topics[:3]]
        if len(senses) >= 2:
            opts = ", ".join(senses[:-1]) + f", or {senses[-1]}"
            return f"That could mean a few things — did you mean the {opts} sense?"
        return "Could you add a bit more detail so I answer the right thing?"

    # ---- convenience: run to completion against a simulated user (for eval) ----
    def resolve(self, query: str, intended: Doc, max_turns: int = 3) -> "Resolution":
        """Drive the conversation until an answer, simulating a user who, when
        asked, supplies the disambiguating hint for their intended sense."""
        q, asks = query, 0
        for _ in range(max_turns):
            turn = self.step(q)
            if turn.action == "answer":
                return Resolution(turn.doc, asks, correct=turn.doc is intended)
            asks += 1
            if intended.hint:
                q = f"{q} {intended.hint}"
            else:
                break
        # forced best-effort answer
        hits = self.retriever.search(q, k=1)
        doc = hits[0].doc if hits else None
        return Resolution(doc, asks, correct=doc is intended)


@dataclass
class Resolution:
    doc: Optional[Doc]
    asks: int
    correct: bool
