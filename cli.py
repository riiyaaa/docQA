"""Command line entry point.

  python cli.py ingest ./my_docs          # index a folder or files
  python cli.py ask "What is the refund policy?"
  python cli.py chat                      # interactive loop
"""
import argparse

from docqa.app import make_answerer, make_vectorstore
from docqa.config import settings
from docqa.ingest import ingest


def print_result(result: dict) -> None:
    print("\n" + result["answer"])
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['source']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG document Q&A")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("paths", nargs="+")
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    sub.add_parser("chat")
    args = parser.parse_args()

    if args.cmd == "ingest":
        n = ingest(args.paths, make_vectorstore(), settings.chunk_size, settings.chunk_overlap)
        print(f"Indexed {n} chunks into {settings.persist_dir}")
        return

    answerer = make_answerer()
    if args.cmd == "ask":
        print_result(answerer.ask(args.question))
        return

    print("Ask a question (Ctrl+C to quit).")
    try:
        while True:
            q = input("\n> ").strip()
            if q:
                print_result(answerer.ask(q))
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
