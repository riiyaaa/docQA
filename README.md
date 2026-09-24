# DocQA: Local RAG Document Q&A

Ask questions about your own PDFs, Word files, and text notes. Answers cite the
exact passage and page they came from, and the bot says so when the answer
isn't in your documents. Everything runs locally with Ollama: no API keys, no cost.

## How it works

1. **Ingest:** files are split into ~800-character overlapping chunks, embedded
   with `nomic-embed-text`, and stored in Chroma.
2. **Retrieve:** each question runs through vector search (meaning) and BM25
   keyword search (exact terms), merged with Reciprocal Rank Fusion.
3. **Answer:** `llama3.2:3b` answers from the top passages only, with
   `[n]` citations. Invalid citations are filtered out.

## Setup

```bash
# 1. Install Ollama from https://ollama.com, then pull the models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Install Python dependencies (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Index documents and ask
python cli.py ingest ./my_docs
python cli.py ask "What does the warranty cover?"
python cli.py chat
```

## Tests

`pytest` runs the pipeline with fake embeddings and a scripted LLM, so no
Ollama is needed.
