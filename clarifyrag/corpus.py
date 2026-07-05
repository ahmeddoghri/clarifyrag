"""A tiny document store with an interpretable TF-IDF-ish retriever.

Deliberately seeded with polysemous topics (apple, jaguar, python, mercury) so
ambiguous queries retrieve *conflicting* senses — which is exactly the signal
ClarifyRAG uses to decide when to ask a clarifying question.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Doc:
    id: str
    topic: str          # the sense/topic this doc belongs to
    hint: str           # the word a user would add to disambiguate to this sense
    text: str
    answer: str


DEFAULT_CORPUS: list[Doc] = [
    Doc("apple_inc", "company", "company",
        "Apple Inc is a technology company; its stock trades on NASDAQ with a large market cap and share price.",
        "Apple Inc. (AAPL) is a tech company; recent share price is around $190."),
    Doc("apple_fruit", "fruit", "fruit",
        "Apple is a fruit that grows on trees; nutrition, varieties, and price per pound at the grocery.",
        "An apple is a fruit; roughly $2 per pound at the grocery store."),
    Doc("jaguar_car", "car", "car",
        "Jaguar is a British luxury car brand; its models, horsepower, and price are well known.",
        "Jaguar is a British luxury car brand (e.g. the F-Type)."),
    Doc("jaguar_animal", "animal", "animal",
        "The jaguar is a big cat native to the Americas; its habitat, diet, and behavior.",
        "The jaguar is a large wild cat native to the Americas."),
    Doc("python_lang", "language", "programming",
        "Python is a programming language; its syntax, libraries, and versions are widely used.",
        "Python is a popular high-level programming language."),
    Doc("python_snake", "snake", "snake",
        "The python is a large non-venomous snake; species, habitat, and how fast it moves.",
        "A python is a large non-venomous constrictor snake."),
    Doc("mercury_planet", "planet", "planet",
        "Mercury is the closest planet to the Sun; its orbit and surface temperature.",
        "Mercury is the smallest planet and the closest to the Sun."),
    Doc("mercury_element", "element", "element",
        "Mercury is a liquid metal element, symbol Hg; its toxicity and industrial uses.",
        "Mercury (Hg) is a liquid metallic chemical element."),
    # unambiguous factual docs
    Doc("eiffel", "landmark", "",
        "The Eiffel Tower in Paris is 330 meters tall and made of iron.",
        "The Eiffel Tower is 330 meters tall."),
    Doc("light", "physics", "",
        "The speed of light in a vacuum is 299792458 meters per second.",
        "The speed of light is about 299,792,458 m/s."),
    Doc("water", "chemistry", "",
        "Water boils at 100 degrees Celsius at sea level atmospheric pressure.",
        "Water boils at 100°C at sea level."),
]


@dataclass
class Hit:
    doc: Doc
    score: float


class Retriever:
    """Interpretable TF-IDF retrieval over the corpus."""

    def __init__(self, docs: Optional[Sequence[Doc]] = None) -> None:
        self.docs = list(docs if docs is not None else DEFAULT_CORPUS)
        n = len(self.docs)
        df: dict[str, int] = {}
        self._doc_tokens: dict[str, set[str]] = {}
        for d in self.docs:
            toks = set(tokenize(d.text)) | set(tokenize(d.id.replace("_", " ")))
            self._doc_tokens[d.id] = toks
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = set(tokenize(query))
        hits: list[Hit] = []
        for d in self.docs:
            shared = q & self._doc_tokens[d.id]
            score = sum(self.idf.get(t, 0.0) for t in shared)
            if score > 0:
                hits.append(Hit(d, round(score, 4)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
