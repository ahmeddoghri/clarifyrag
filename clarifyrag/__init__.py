"""ClarifyRAG — a clarification-aware search agent that asks before it answers.

>>> from clarifyrag import ClarifyRAG
>>> agent = ClarifyRAG()
>>> agent.step("tell me about jaguar").action
'ask'
>>> agent.step("jaguar animal wild cat habitat").action
'answer'
"""
from .agent import ClarifyRAG, Resolution, Turn
from .corpus import DEFAULT_CORPUS, Doc, Hit, Retriever, tokenize
from .gate import GateDecision, assess

__all__ = [
    "ClarifyRAG", "Turn", "Resolution",
    "Retriever", "Doc", "Hit", "DEFAULT_CORPUS", "tokenize",
    "GateDecision", "assess",
]
__version__ = "0.1.0"
