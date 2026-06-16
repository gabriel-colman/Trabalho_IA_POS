"""
Coleta suplementar via OpenAlex para completar corpus até 1000+ artigos.
Usa termos mais diretos de áudio+ML que a coleta anterior não cobriu.
"""
import json, time, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import requests

OPENALEX_URL = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "IA-FACOM-UFMS/2026 (mailto:gabrielcolman65@gmail.com)"}

EXTRA_TERMS = [
    "bioacoustic monitoring machine learning",
    "soundscape ecology deep learning",
    "acoustic monitoring bird species",
    "audio species identification deep learning",
    "bird audio deep learning",
    "environmental audio classification",
    "acoustic bird monitoring neural",
    "wildlife sound deep learning",
    "bird call deep learning",
    "bird vocalization deep learning",
]

YEAR_FROM, YEAR_TO = 2016, 2026
MAX_PER_TERM = 300

SUPPLEMENT_PATH = Path("data/openalex_supplement.jsonl")


def normalize_abstract(inv_index):
    if not inv_index: return ""
    words = {}
    for word, positions in inv_index.items():
        for pos in positions: words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def fetch_term(term, year_from, year_to, max_results):
    collected = []
    params = {
        "search": term,
        "filter": f"publication_year:{year_from}-{year_to},has_abstract:true",
        "select": "id,doi,title,abstract_inverted_index,authorships,primary_topic,topics,publication_year",
        "per-page": 200, "cursor": "*",
    }
    while len(collected) < max_results:
        try:
            r = requests.get(OPENALEX_URL, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    Erro: {e}"); break
        results = data.get("results", [])
        if not results: break
        for work in results:
            doc_id = (work.get("id") or "").replace("https://openalex.org/", "")
            if not doc_id: continue
            abstract = normalize_abstract(work.get("abstract_inverted_index"))
            title = (work.get("title") or "").strip()
            if not title or len(abstract) < 50: continue
            authors = [a["author"]["display_name"] for a in (work.get("authorships") or []) if a.get("author")]
            topic = work.get("primary_topic") or {}
            categories = [t.get("display_name", "") for t in (work.get("topics") or [])]
            collected.append({
                "arxiv_id": doc_id, "title": title, "abstract": abstract,
                "authors": authors, "categories": categories,
                "primary_category": topic.get("display_name", ""),
                "published": str(work.get("publication_year", "")),
                "doi": work.get("doi"), "source": "openalex",
            })
            if len(collected) >= max_results: break
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor: break
        params["cursor"] = next_cursor
        time.sleep(0.5)
    return collected


# Carrega IDs já no openalex_raw
existing_ids = set()
for path in [Path("data/openalex_raw.jsonl"), SUPPLEMENT_PATH]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try: existing_ids.add(json.loads(line).get("arxiv_id", ""))
                except: pass
print(f"IDs existentes: {len(existing_ids)}")

new_records = []
for term in EXTRA_TERMS:
    print(f"Buscando: {term!r} ...")
    results = fetch_term(term, YEAR_FROM, YEAR_TO, MAX_PER_TERM)
    added = 0
    for rec in results:
        if rec["arxiv_id"] not in existing_ids:
            existing_ids.add(rec["arxiv_id"])
            new_records.append(rec)
            added += 1
    print(f"  -> {added} novos (total novo={len(new_records)})")
    time.sleep(1)

with open(SUPPLEMENT_PATH, "w", encoding="utf-8") as f:
    for rec in new_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\nSalvos {len(new_records)} registros em {SUPPLEMENT_PATH}")
