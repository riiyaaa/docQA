"""Phase 2: measure retrieval and answer quality against a labeled question set.

Everything here takes its models and retriever as arguments, so the tests can
run it with fake models. eval/run_eval.py wires in the real Ollama models.
"""
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .generate import NOT_FOUND
from .retrieve import reciprocal_rank_fusion

RETRIEVAL_MODES = ("vector", "bm25", "hybrid")

# Phrases small models use when they refuse in their own words instead of the
# exact NOT_FOUND sentence. Counted separately so paraphrased refusals show up.
REFUSAL_PHRASES = (
    "couldn't find", "could not find", "can't find", "cannot find",
    "not mentioned", "no information", "not in the documents", "not provided",
    "does not contain", "doesn't contain", "do not contain", "don't contain",
    "not specified", "not stated", "no mention",
    "don't say", "doesn't say", "do not say", "does not say",
)

# Citations like [1, 2] or [1-3] that the answer step's [n] parser misses.
UNPARSED_CITATION = re.compile(r"\[\s*\d+\s*(?:[,;\-–]\s*\d+\s*)+\]")


# ---------- loading ----------

def load_questions(path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def gold_pages(question: dict) -> set:
    """The (file, page) pairs where the answer lives. page is None for files without pages."""
    return {(s["source"], s.get("page")) for s in question["sources"]}


def doc_page(doc) -> tuple:
    return (doc.metadata["source"], doc.metadata.get("page"))


# ---------- scoring helpers ----------

def is_exact_refusal(answer: str) -> bool:
    """True if the answer contains the fixed refusal sentence."""
    return NOT_FOUND.lower().rstrip(".") in answer.lower()


def is_refusal(answer: str) -> bool:
    """True for the exact refusal or a paraphrase of it."""
    text = answer.lower()
    return is_exact_refusal(answer) or any(p in text for p in REFUSAL_PHRASES)


def keywords_match(answer: str, keyword_groups: list[str]) -> bool:
    """Every group must match. Inside a group, '|' separates acceptable alternatives."""
    text = answer.lower()
    return all(any(alt.strip().lower() in text for alt in group.split("|"))
               for group in keyword_groups)


def first_hit_rank(docs, gold: set):
    """1-based position of the first retrieved chunk from a gold page, or None."""
    for i, doc in enumerate(docs, start=1):
        if doc_page(doc) in gold:
            return i
    return None


# ---------- retrieval ----------

def evaluate_retrieval(questions, retriever) -> list[dict]:
    """Rank each answerable question with vector search, BM25, and hybrid fusion.
    No LLM is involved, so this part is fast."""
    rows = []
    for q in questions:
        if q["type"] == "unanswerable":
            continue
        gold = gold_pages(q)
        vec = retriever.vector_search(q["question"])
        kw = retriever.keyword_search(q["question"])
        rankings = {"vector": vec, "bm25": kw, "hybrid": reciprocal_rank_fusion([vec, kw])}
        row = {"id": q["id"], "type": q["type"]}
        for mode, docs in rankings.items():
            row[f"{mode}_rank"] = first_hit_rank(docs[:10], gold)
        rows.append(row)
    return rows


def summarize_retrieval(rows, top_k: int) -> dict:
    """hit@1, hit@top_k (what the LLM actually sees), hit@10 (the candidate pool a
    reranker could draw from), and MRR, per mode, overall and by question type."""
    def stats(subset):
        out = {"n": len(subset)}
        for mode in RETRIEVAL_MODES:
            ranks = [r[f"{mode}_rank"] for r in subset]
            n = len(ranks) or 1
            out[mode] = {
                "hit@1": sum(1 for k in ranks if k == 1) / n,
                f"hit@{top_k}": sum(1 for k in ranks if k and k <= top_k) / n,
                "hit@10": sum(1 for k in ranks if k) / n,
                "mrr": sum(1 / k for k in ranks if k) / n,
            }
        return out

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {"overall": stats(rows), "by_type": {t: stats(rs) for t, rs in sorted(by_type.items())}}


# ---------- answers ----------

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You grade answers from a document question-answering system. Reply with JSON only."),
    ("human", """Question: {question}

Reference answer: {reference}

Passages the system was given:
{passages}

System answer: {answer}

Grade two things:
- "correct": true if the system answer states the same key facts as the reference answer. Wording may differ; extra correct detail is fine.
- "grounded": true if every factual claim in the system answer is supported by the passages.

Reply exactly in this form: {{"correct": true, "grounded": true, "reason": "one short sentence"}}"""),
])


class Judge:
    """LLM-as-judge. A small local model is a rough grader: spot-check its verdicts."""

    def __init__(self, llm):
        self.chain = JUDGE_PROMPT | llm | StrOutputParser()

    def grade(self, question: str, reference: str, answer: str, docs) -> dict:
        passages = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1))
        raw = self.chain.invoke({"question": question, "reference": reference,
                                 "passages": passages, "answer": answer})
        return parse_judgement(raw)


def parse_judgement(raw: str) -> dict:
    """Read the judge's JSON, tolerating extra text around it."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data.get("correct"), bool) or not isinstance(data.get("grounded"), bool):
        return {"correct": None, "grounded": None, "reason": f"Unreadable judge output: {raw[:120]}"}
    return {"correct": data["correct"], "grounded": data["grounded"], "reason": str(data.get("reason", ""))}


def score_answer(question: dict, result: dict) -> dict:
    """Rule-based checks that need no extra model."""
    answer = result["answer"]
    refused = is_refusal(answer)
    cited = {(s["file"], s["page"]) for s in result["sources"]}
    row = {
        "refused": refused,
        "refused_exact": is_exact_refusal(answer),
        "unparsed_citation": bool(UNPARSED_CITATION.search(answer)),
        "cited": sorted(cited, key=str),
    }
    if question["type"] == "unanswerable":
        row["correct"] = refused
    else:
        row["correct"] = (not refused) and keywords_match(answer, question["answer_keywords"])
        row["has_citation"] = bool(cited)
        row["cited_gold"] = bool(cited & gold_pages(question))
    return row


def evaluate_answers(questions, retriever, answerer, judge=None, progress=print) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        docs = retriever.retrieve(q["question"])
        t1 = time.perf_counter()
        result = answerer.answer(q["question"], docs)
        t2 = time.perf_counter()

        row = {"id": q["id"], "type": q["type"], "question": q["question"],
               "expected": q["answer"], "answer": result["answer"],
               "retrieved": [list(doc_page(d)) for d in docs],
               "retrieval_seconds": round(t1 - t0, 3), "answer_seconds": round(t2 - t1, 3)}
        if q["type"] != "unanswerable":
            row["retrieval_hit"] = first_hit_rank(docs, gold_pages(q)) is not None
        row.update(score_answer(q, result))

        # The judge only grades real answers to answerable questions; a refusal
        # there is already wrong, and refusals to unanswerable ones are checked above.
        if judge and q["type"] != "unanswerable" and not row["refused"]:
            row.update({f"judge_{k}": v for k, v in
                        judge.grade(q["question"], q["answer"], result["answer"], docs).items()})

        progress(f"[{i:>2}/{len(questions)}] {q['id']} {q['type']:<12} "
                 f"{'PASS' if row['correct'] else 'FAIL'}  ({row['answer_seconds']:.1f}s)")
        rows.append(row)
    return rows


def summarize_answers(rows) -> dict:
    def rate(items, key):
        values = [r[key] for r in items if r.get(key) is not None]
        return (sum(values) / len(values)) if values else None

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in answerable if not r["refused"]]

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    summary = {
        "n": len(rows),
        "accuracy_answerable": rate(answerable, "correct"),
        "retrieval_hit": rate(answerable, "retrieval_hit"),
        "false_refusal_rate": rate(answerable, "refused"),
        "refusal_accuracy": rate(unanswerable, "refused"),
        "refusal_exact_wording": rate(unanswerable, "refused_exact"),
        "citation_rate": rate(answered, "has_citation"),
        "cited_correct_page": rate(answered, "cited_gold"),
        "unparsed_citations": sum(r["unparsed_citation"] for r in rows),
        "avg_answer_seconds": (sum(r["answer_seconds"] for r in rows) / len(rows)) if rows else None,
        "by_type": {t: {"n": len(rs), "accuracy": rate(rs, "correct")} for t, rs in sorted(by_type.items())},
    }
    if any("judge_correct" in r for r in rows):
        judged = [r for r in rows if "judge_correct" in r]
        summary["judge"] = {
            "n": len(judged),
            "correct": rate(judged, "judge_correct"),
            "grounded": rate(judged, "judge_grounded"),
            "unreadable": sum(1 for r in judged if r["judge_correct"] is None),
        }
    return summary


# ---------- report ----------

def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def format_report(config: dict, retrieval: dict, answers: dict | None) -> str:
    k = config["top_k"]
    lines = ["# DocQA evaluation", ""]
    lines += [f"- {key}: `{value}`" for key, value in config.items()]
    lines += ["", f"## Retrieval ({retrieval['overall']['n']} answerable questions)", "",
              f"| Mode | hit@1 | hit@{k} | hit@10 | MRR |", "|---|---|---|---|---|"]
    for mode in RETRIEVAL_MODES:
        s = retrieval["overall"][mode]
        lines.append(f"| {mode} | {pct(s['hit@1'])} | {pct(s[f'hit@{k}'])} | {pct(s['hit@10'])} | {s['mrr']:.2f} |")
    lines += ["", f"hit@{k} by question type:", "",
              "| Type | n | vector | bm25 | hybrid |", "|---|---|---|---|---|"]
    for t, s in retrieval["by_type"].items():
        lines.append(f"| {t} | {s['n']} | " + " | ".join(pct(s[m][f"hit@{k}"]) for m in RETRIEVAL_MODES) + " |")

    if answers:
        a = answers
        lines += ["", f"## Answers ({a['n']} questions)", "", "| Metric | Value |", "|---|---|",
                  f"| Accuracy on answerable questions (keyword check) | {pct(a['accuracy_answerable'])} |",
                  f"| Right page retrieved (hybrid, top {k}) | {pct(a['retrieval_hit'])} |",
                  f"| Wrongly refused an answerable question | {pct(a['false_refusal_rate'])} |",
                  f"| Refused unanswerable questions | {pct(a['refusal_accuracy'])} |",
                  f"| ...using the exact refusal sentence | {pct(a['refusal_exact_wording'])} |",
                  f"| Answers with at least one citation | {pct(a['citation_rate'])} |",
                  f"| Answers citing a correct page | {pct(a['cited_correct_page'])} |",
                  f"| Answers with unparsed citations like [1, 2] | {a['unparsed_citations']} |",
                  f"| Average answer time | {a['avg_answer_seconds']:.1f}s |"]
        if "judge" in a:
            j = a["judge"]
            lines += [f"| Judge: correct ({j['n']} judged) | {pct(j['correct'])} |",
                      f"| Judge: grounded in passages | {pct(j['grounded'])} |",
                      f"| Judge: unreadable verdicts | {j['unreadable']} |"]
        lines += ["", "Accuracy by question type:", "", "| Type | n | accuracy |", "|---|---|---|"]
        lines += [f"| {t} | {s['n']} | {pct(s['accuracy'])} |" for t, s in a["by_type"].items()]
    return "\n".join(lines) + "\n"
