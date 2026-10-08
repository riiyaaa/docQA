"""Run the DocQA evaluation on the sample documents.

    python eval/run_eval.py --retrieval-only          # retrieval only, about a minute
    python eval/run_eval.py                           # retrieval + answers (several minutes on CPU)
    python eval/run_eval.py --judge                   # also grade answers with an LLM judge
    python eval/run_eval.py --limit 5                 # quick smoke test on 5 questions

The documents are indexed into a fresh in-memory collection on every run, so
results never depend on what is in your chroma_db folder. Each run saves a JSON
file and a Markdown report in eval/results/.
"""
import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_chroma import Chroma  # noqa: E402
from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402

from docqa.config import settings  # noqa: E402
from docqa.evaluate import (Judge, evaluate_answers, evaluate_retrieval, format_report,  # noqa: E402
                            load_questions, summarize_answers, summarize_retrieval)
from docqa.generate import Answerer  # noqa: E402
from docqa.ingest import ingest  # noqa: E402
from docqa.retrieve import HybridRetriever  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate DocQA retrieval and answers.")
    p.add_argument("--docs", default=str(ROOT / "sample_docs"))
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    p.add_argument("--out", default=str(ROOT / "eval" / "results"))
    p.add_argument("--retrieval-only", action="store_true", help="skip answer generation (fast)")
    p.add_argument("--judge", action="store_true", help="grade answers with an LLM judge")
    p.add_argument("--judge-model", default=settings.llm_model,
                   help="Ollama model for the judge; a larger one (e.g. qwen2.5:7b) grades more reliably")
    p.add_argument("--llm", default=settings.llm_model)
    p.add_argument("--embed", default=settings.embed_model)
    p.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    p.add_argument("--candidate-k", type=int, default=settings.candidate_k)
    p.add_argument("--top-k", type=int, default=settings.top_k)
    p.add_argument("--limit", type=int, help="only use the first N questions")
    p.add_argument("--types", help="comma-separated question types to include, e.g. reworded,exact_term")
    p.add_argument("--label", default="", help="short name for this run, added to the file names")
    args = p.parse_args()

    questions = load_questions(args.questions)
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        questions = [q for q in questions if q["type"] in wanted]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        sys.exit("No questions selected.")

    config = {
        "llm": args.llm, "embed": args.embed, "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap, "candidate_k": args.candidate_k, "top_k": args.top_k,
        "questions": len(questions), "commit": git_commit(),
    }
    if args.judge and not args.retrieval_only:
        config["judge"] = args.judge_model

    print(f"Indexing {args.docs} ...")
    embeddings = OllamaEmbeddings(model=args.embed, base_url=settings.ollama_url)
    store = Chroma(collection_name=f"eval-{uuid.uuid4().hex[:8]}", embedding_function=embeddings)
    n_chunks = ingest([args.docs], store, args.chunk_size, args.chunk_overlap)
    config["chunks"] = n_chunks
    retriever = HybridRetriever(store, args.candidate_k, args.top_k)

    print(f"Retrieval: {len(questions)} questions ...")
    retrieval = summarize_retrieval(evaluate_retrieval(questions, retriever), args.top_k)

    answer_rows, answers = None, None
    if not args.retrieval_only:
        print("Answers (the first one is slow while the model loads):")
        llm = ChatOllama(model=args.llm, base_url=settings.ollama_url, temperature=0)
        judge = None
        if args.judge:
            judge = Judge(ChatOllama(model=args.judge_model, base_url=settings.ollama_url,
                                     temperature=0, format="json"))
        answer_rows = evaluate_answers(questions, retriever, Answerer(llm, retriever), judge)
        answers = summarize_answers(answer_rows)

    report = format_report(config, retrieval, answers)
    print("\n" + report)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{stamp}" + (f"_{args.label}" if args.label else "")
    (out / f"{name}.md").write_text(report, encoding="utf-8")
    (out / f"{name}.json").write_text(json.dumps(
        {"config": config, "retrieval": retrieval, "answers": answers, "rows": answer_rows},
        indent=2), encoding="utf-8")
    print(f"Saved {out / name}.md and .json")


if __name__ == "__main__":
    main()
