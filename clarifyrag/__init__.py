"""ClarifyRAG — a clarification-aware search agent that asks before it answers.

>>> from clarifyrag import ClarifyRAG
>>> agent = ClarifyRAG()
>>> agent.step("tell me about jaguar").action
'ask'
>>> agent.step("jaguar animal wild cat habitat").action
'answer'
"""
from .corpus import Doc, Hit, Retriever, DEFAULT_CORPUS, tokenize
from .gate import GateDecision, assess
from .agent import ClarifyRAG, Turn, Resolution

__all__ = [
    "ClarifyRAG", "Turn", "Resolution",
    "Retriever", "Doc", "Hit", "DEFAULT_CORPUS", "tokenize",
    "GateDecision", "assess",
]
__version__ = "0.1.0"
