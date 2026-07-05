from clarifyrag import ClarifyRAG, Retriever, assess
from clarifyrag.corpus import DEFAULT_CORPUS
from clarifyrag.eval import run, BENCH


def _doc(id):
    return next(d for d in DEFAULT_CORPUS if d.id == id)


def test_ambiguous_query_triggers_ask():
    agent = ClarifyRAG()
    turn = agent.step("tell me about jaguar")
    assert turn.action == "ask"
    assert turn.gate.ambiguous


def test_specified_query_answers_directly():
    agent = ClarifyRAG()
    turn = agent.step("jaguar animal wild cat habitat")
    assert turn.action == "answer"
    assert turn.doc.id == "jaguar_animal"


def test_unambiguous_fact_answers():
    agent = ClarifyRAG()
    turn = agent.step("boiling point of water at sea level")
    assert turn.action == "answer"
    assert turn.doc.id == "water"


def test_clarification_resolves_to_intended_sense():
    agent = ClarifyRAG()
    res = agent.resolve("how fast is python", _doc("python_snake"))
    assert res.correct
    assert res.asks >= 1  # it had to ask to get there


def test_gate_has_strong_f1_on_bench():
    (prec, rec, f1), _ = run()
    assert f1 >= 0.8
    assert prec >= 0.8 and rec >= 0.8


def test_clarify_aware_beats_naive_policies():
    _, policies = run()
    search_util = policies["always_search"][2]
    ask_util = policies["always_ask"][2]
    clarify_util = policies["clarify_aware"][2]
    assert clarify_util > search_util
    assert clarify_util > ask_util


def test_assess_reports_contending_topics():
    d = assess("what is the price of apple", Retriever())
    assert d.ambiguous
    assert len(d.topics) >= 2
