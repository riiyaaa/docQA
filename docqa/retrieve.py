"""Step 2: hybrid retrieval.

Vector search finds passages with similar *meaning* ("car won't start" matches
"engine failure"). Keyword search (BM25) finds exact *terms* that embeddings
often blur: part numbers, names, acronyms. We run both and merge the rankings
with Reciprocal Rank Fusion (RRF).
"""
import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge several ranked lists. Each document scores 1 / (k + rank) in every
    list it appears in, and the scores are summed. Ranks are used instead of raw
    scores because vector distances and BM25 scores are on different scales.
    k=60 is the constant from the original RRF paper; it keeps a single #1
    result from dominating."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            cid = doc.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            docs[cid] = doc
    return [docs[cid] for cid in sorted(scores, key=scores.get, reverse=True)]


class HybridRetriever:
    def __init__(self, vectorstore, candidate_k: int = 10, top_k: int = 4):
        self.vectorstore = vectorstore
        self.candidate_k = candidate_k
        self.top_k = top_k
        # Build the BM25 index from the chunks already stored in Chroma, so the
        # vector store stays the single source of truth.
        data = vectorstore.get(include=["documents", "metadatas"])
        self.chunks = [Document(page_content=t, metadata=m)
                       for t, m in zip(data["documents"], data["metadatas"])]
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks]) if self.chunks else None

    def keyword_search(self, query: str) -> list[Document]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Drop zero-score chunks: they share no terms with the query at all
        return [self.chunks[i] for i in ranked[: self.candidate_k] if scores[i] > 0]

    def vector_search(self, query: str) -> list[Document]:
        if not self.chunks:
            return []
        return self.vectorstore.similarity_search(query, k=min(self.candidate_k, len(self.chunks)))

    def retrieve(self, query: str) -> list[Document]:
        fused = reciprocal_rank_fusion([self.vector_search(query), self.keyword_search(query)])
        return fused[: self.top_k]
