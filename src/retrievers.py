"""Retrievers reutilizáveis: BM25, TFIDF-KNN e Híbrido (RRF)."""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import tokenize, doc_text


class BM25Retriever:
    """Recuperador BM25Okapi (paradigma probabilístico)."""

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self._tokenized = [tokenize(doc_text(d)) for d in docs]
        self.index = BM25Okapi(self._tokenized, k1=k1, b=b)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        scores = self.index.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.docs[i]["arxiv_id"], float(scores[i])) for i in top_idx]


class TFIDFKNNRetriever:
    """Recuperador denso: TF-IDF vetorial + similaridade de cosseno."""

    def __init__(self, docs: list[dict], max_features: int = 50_000,
                 ngram_range: tuple = (1, 2)):
        self.docs = docs
        self._vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        texts = [doc_text(d) for d in docs]
        self._matrix = self._vec.fit_transform(texts)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        q_vec = self._vec.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:k]
        return [(self.docs[i]["arxiv_id"], float(sims[i])) for i in top_idx]


class HybridRRFRetriever:
    """
    M5 — Ranking Híbrido via Reciprocal Rank Fusion (RRF).

    Combina listas de BM25 e TF-IDF-KNN sem normalização de scores:
    RRF_score(d) = sum_r  1 / (k + rank_r(d))
    onde k=60 é a constante padrão da literatura (Cormack et al. 2009).
    """

    def __init__(self, docs: list[dict],
                 bm25_k1: float = 1.5, bm25_b: float = 0.75,
                 tfidf_features: int = 50_000,
                 rrf_k: int = 60):
        self.bm25 = BM25Retriever(docs, k1=bm25_k1, b=bm25_b)
        self.tfidf = TFIDFKNNRetriever(docs, max_features=tfidf_features)
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 100,
               pool_size: int = 1000) -> list[tuple[str, float]]:
        bm25_ranked = self.bm25.search(query, k=pool_size)
        tfidf_ranked = self.tfidf.search(query, k=pool_size)

        rrf: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(bm25_ranked, 1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (doc_id, _) in enumerate(tfidf_ranked, 1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
