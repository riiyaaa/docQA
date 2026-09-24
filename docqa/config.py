"""All tunable settings in one place. Override any of them with environment variables."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Models served by a local Ollama instance
    llm_model: str = os.getenv("DOCQA_LLM", "llama3.2:3b")
    embed_model: str = os.getenv("DOCQA_EMBED", "nomic-embed-text")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Vector store location
    persist_dir: str = os.getenv("DOCQA_DB", "./chroma_db")
    collection: str = "docqa"

    # Chunking: ~800 characters is roughly 150-200 words, small enough to be
    # precise, large enough to keep a full thought together. Overlap stops an
    # answer from being cut in half at a chunk boundary.
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval: each search method returns `candidate_k` hits, fusion keeps `top_k`
    candidate_k: int = 10
    top_k: int = 4


settings = Settings()
