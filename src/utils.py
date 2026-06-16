"""Funções utilitárias compartilhadas entre notebooks."""
import json
from pathlib import Path

import pandas as pd


def load_corpus(path: str | Path = "data/corpus.jsonl") -> list[dict]:
    """Carrega o corpus em lista de dicts."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
    return docs


def load_queries(path: str | Path = "eval/queries.tsv") -> pd.DataFrame:
    """Carrega queries no formato qid<TAB>texto."""
    return pd.read_csv(path, sep="\t", names=["qid", "text"], dtype=str)


def write_trec_run(results: dict[str, list[tuple[str, float]]],
                   run_path: str | Path,
                   system_name: str = "system") -> None:
    """
    Salva resultados em formato TREC.

    results: {qid: [(doc_id, score), ...]} já ordenado por score desc
    """
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, ranked in results.items():
            for rank, (doc_id, score) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {system_name}\n")
