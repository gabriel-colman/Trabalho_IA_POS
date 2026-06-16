"""Gera os 3 arquivos .trec com o corpus atual."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_corpus, load_queries, write_trec_run
from src.retrievers import BM25Retriever, TFIDFKNNRetriever, HybridRRFRetriever

docs = load_corpus("data/corpus.jsonl")
queries = load_queries("eval/queries.tsv")
runs_dir = Path("notebooks/runs")
runs_dir.mkdir(parents=True, exist_ok=True)

print(f"Corpus: {len(docs)} docs | Queries: {len(queries)}")

print("Construindo BM25...")
bm25 = BM25Retriever(docs, k1=1.5, b=0.75)
print("Construindo TF-IDF KNN...")
tfidf = TFIDFKNNRetriever(docs, max_features=50_000, ngram_range=(1, 2))
print("Construindo Híbrido RRF (k=60)...")
hybrid = HybridRRFRetriever(docs, rrf_k=60)

for name, ret in [("bm25", bm25), ("knn_tfidf", tfidf), ("hybrid_rrf", hybrid)]:
    results = {row["qid"]: ret.search(row["text"], k=100)
               for _, row in queries.iterrows()}
    out = runs_dir / f"{name}.trec"
    write_trec_run(results, out, system_name=name)
    print(f"Salvo: {out}")
