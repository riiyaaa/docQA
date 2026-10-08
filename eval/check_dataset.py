"""Check that eval/questions.jsonl matches the documents in sample_docs/.

For every answerable question, each evidence phrase must appear in at least
one of its listed sources (same file and page). Run it after editing the
documents or the questions:

    python eval/check_dataset.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docqa.generate import NOT_FOUND  # noqa: E402
from docqa.ingest import collect_files, load_file  # noqa: E402

DOCS = ROOT / "sample_docs"
QUESTIONS = ROOT / "eval" / "questions.jsonl"
TYPES = {"easy", "exact_term", "reworded", "multi_part", "unanswerable"}
FIELDS = {"id", "type", "question", "answer", "answer_keywords", "sources", "evidence"}


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, so line breaks inside PDFs don't matter."""
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    # (file name, page) -> text, exactly as the ingest step reads it
    pages = {}
    for f in collect_files([str(DOCS)]):
        for d in load_file(f):
            pages[(d.metadata["source"], d.metadata.get("page"))] = normalize(d.page_content)

    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []
    ids = Counter(r.get("id") for r in rows)

    for r in rows:
        qid = r.get("id", "?")
        if missing := FIELDS - r.keys():
            errors.append(f"{qid}: missing fields {sorted(missing)}")
            continue
        if ids[qid] > 1:
            errors.append(f"{qid}: duplicate id")
        if r["type"] not in TYPES:
            errors.append(f"{qid}: unknown type '{r['type']}'")

        if r["type"] == "unanswerable":
            if r["answer"] != NOT_FOUND or r["sources"] or r["evidence"]:
                errors.append(f"{qid}: unanswerable questions need answer = NOT_FOUND and no sources/evidence")
            continue

        if not r["sources"] or not r["evidence"]:
            errors.append(f"{qid}: answerable questions need sources and evidence")
            continue
        texts = []
        for s in r["sources"]:
            key = (s["source"], s.get("page"))
            if key not in pages:
                errors.append(f"{qid}: source not found in sample_docs: {key}")
            else:
                texts.append(pages[key])
        for phrase in r["evidence"]:
            if texts and not any(normalize(phrase) in t for t in texts):
                errors.append(f"{qid}: evidence not found in listed sources: '{phrase}'")

    counts = Counter(r.get("type") for r in rows)
    print(f"{len(rows)} questions: " + ", ".join(f"{t} {counts[t]}" for t in sorted(counts)))
    print(f"{len(pages)} document sections read from {DOCS.name}/")
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print("  - " + e)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
