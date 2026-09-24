"""Wires the real components together (Ollama + Chroma)."""
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import settings
from .generate import Answerer
from .retrieve import HybridRetriever


def make_vectorstore(embeddings=None):
    embeddings = embeddings or OllamaEmbeddings(
        model=settings.embed_model, base_url=settings.ollama_url)
    return Chroma(collection_name=settings.collection, embedding_function=embeddings,
                  persist_directory=settings.persist_dir)


def make_answerer(vectorstore=None, llm=None) -> Answerer:
    vectorstore = vectorstore or make_vectorstore()
    # temperature=0 makes answers repeatable, which matters for evaluation later
    llm = llm or ChatOllama(model=settings.llm_model, base_url=settings.ollama_url, temperature=0)
    retriever = HybridRetriever(vectorstore, settings.candidate_k, settings.top_k)
    return Answerer(llm, retriever)
