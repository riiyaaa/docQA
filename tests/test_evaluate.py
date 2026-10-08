"""Tests for the evaluation code, using fake models (no Ollama needed)."""
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from docqa.evaluate import (Judge, evaluate_answers, evaluate_retrieval, first_hit_rank, format_report,
                            is_exact_refusal, is_refusal, keywords_match, load_questions,
                            parse_judgement, summarize_answers, summarize_retrieval)
from docqa.generate import NOT_FOUND, Answerer
from docqa.ingest import ingest
from docqa.retrieve import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


def doc(source, page=None):
    meta = {"source": source, "chunk_id": f"{source}:{page}"}
    if page is not None:
        meta["page"] = page
    return Document(page_content="text", metadata=meta)


def test_refusal_detection():
    assert is_exact_refusal(NOT_FOUND)
    assert is_exact_refusal("I couldn't find that in the documents [1].")
    assert not is_exact_refusal("The documents don't mention this.")
    assert is_refusal("The passages do not contain that information.")
    assert not is_refusal("Refunds take 14 business days [1].")


def test_keyword_groups_and_alternatives():
    assert keywords_match("Coverage lasts two years.", ["2 years|two years"])
    assert keywords_match("Hold Mode for 3 seconds.", ["Mode", "3 seconds|three seconds"])
    assert not keywords_match("Hold Mode briefly.", ["Mode", "3 seconds|three seconds"])
    assert keywords_match("Anything.", [])


def test_first_hit_rank_matches_file_and_page():
    docs = [doc("a.pdf", 1), doc("b.pdf", 2), doc("notes.md")]
    assert first_hit_rank(docs, {("b.pdf", 2)}) == 2
    assert first_hit_rank(docs, {("notes.md", None)}) == 3
    assert first_hit_rank(docs, {("b.pdf", 3)}) is None  # right file, wrong page is a miss


def test_parse_judgement_is_tolerant():
    assert parse_judgement('{"correct": true, "grounded": false, "reason": "x"}')["grounded"] is False
    assert parse_judgement('Sure! {"correct": false, "grounded": true}')["correct"] is False
    assert parse_judgement("no json here")["correct"] is None
    assert parse_judgement('{"correct": "yes", "grounded": true}')["correct"] is None


def test_summarize_retrieval_counts_hits():
    rows = [{"id": "a", "type": "easy", "vector_rank": 1, "bm25_rank": None, "hybrid_rank": 2},
            {"id": "b", "type": "easy", "vector_rank": 6, "bm25_rank": 3, "hybrid_rank": None}]
    s = summarize_retrieval(rows, top_k=4)["overall"]
    assert s["vector"]["hit@1"] == 0.5 and s["vector"]["hit@4"] == 0.5 and s["vector"]["hit@10"] == 1.0
    assert s["bm25"]["hit@4"] == 0.5
    assert s["hybrid"]["mrr"] == pytest.approx((1 / 2) / 2)


# ---------- end to end on the real sample documents, with fake models ----------

@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    store = Chroma(collection_name="eval_test", embedding_function=DeterministicFakeEmbedding(size=64),
                   persist_directory=str(tmp_path_factory.mktemp("db")))
    ingest([str(ROOT / "sample_docs")], store, chunk_size=800, chunk_overlap=120)
    questions = load_questions(ROOT / "eval" / "questions.jsonl")
    return HybridRetriever(store, candidate_k=10, top_k=4), questions


def test_retrieval_evaluation_runs_on_sample_docs(setup):
    retriever, questions = setup
    rows = evaluate_retrieval(questions, retriever)
    assert len(rows) == sum(q["type"] != "unanswerable" for q in questions)
    summary = summarize_retrieval(rows, top_k=4)
    # Fake embeddings are random, but BM25 still finds most exact part numbers and codes.
    assert summary["by_type"]["exact_term"]["bm25"]["hit@10"] > 0.5


def test_answer_scoring_end_to_end(setup):
    retriever, questions = setup
    by_id = {q["id"]: q for q in questions}
    picked = [by_id["q12"], by_id["q14"], by_id["q35"]]   # E17 on AP300, ZX-4410, an unanswerable one
    llm = FakeListChatModel(responses=[
        "E17 means the filter life has ended; replace the AP-F300 filter [1, 2].",  # right, but [1, 2] style
        "Yes, ZX-4410 is covered [1].",                                             # wrong answer
        "The documents don't say anything about televisions.",                     # paraphrased refusal
    ])
    judge = Judge(FakeListChatModel(responses=['{"correct": true, "grounded": true, "reason": "ok"}'] * 3))
    rows = evaluate_answers(picked, retriever, Answerer(llm, retriever), judge, progress=lambda _: None)

    e17, zx, tv = rows
    assert e17["correct"] and e17["unparsed_citation"] and not e17["has_citation"]
    assert not zx["correct"]                   # "covered" without "excluded"/"not covered"
    assert tv["correct"] and not tv["refused_exact"]
    assert "judge_correct" in e17 and "judge_correct" not in tv   # refusals are not sent to the judge

    summary = summarize_answers(rows)
    assert summary["unparsed_citations"] == 1
    assert summary["refusal_accuracy"] == 1.0 and summary["refusal_exact_wording"] == 0.0
    report = format_report({"top_k": 4}, summarize_retrieval(evaluate_retrieval(picked, retriever), 4), summary)
    assert "Refused unanswerable questions" in report
