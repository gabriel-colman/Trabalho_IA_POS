"""
Reconstrói o corpus com filtro mais rigoroso.
ArXiv → sempre inclui (todos são CS/ML).
OpenAlex → inclui somente se:
  - categoria primária é explicitamente ML/áudio, OU
  - abstract menciona ML *e* algum termo de áudio/vocalizaçã/bioacústica.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

# Categorias OpenAlex que são explicitamente ML ou processamento de áudio
OA_KEEP_CATS = {
    "cs.SD", "cs.LG", "eess.AS", "eess.SP", "cs.CV", "cs.AI", "cs.NE",
    "Music and Audio Processing",
    "Speech and Audio Processing",
    "Advanced Neural Network Applications",
    "Audio and Speech Processing",
    "Computer Vision and Pattern Recognition",
    "Machine Learning",
    "Acoustic Ecology and Sound Design",
}

# Qualquer menção explícita a bioacústica/vocalizaçã já basta (mesmo sem ML)
CORE_BIO = re.compile(
    r"\b(bioacoustic|ecoacoustic|vocali[sz]|bird.?call|bird.?song|bird.?sound|"
    r"parrot|psittac|avian.?sound|avian.?vocali|wildlife.?sound|wildlife.?audio|"
    r"animal.?call|animal.?vocali|birdnet|birdclef|bird.?species.?identif|"
    r"acoustic.?species|soundscape.?ecolog|passive.?acoustic.?monitor)\b",
    re.IGNORECASE,
)

# Termos de ML
ML_RE = re.compile(
    r"\b(deep learning|machine learning|neural network|convolutional|cnn|rnn|lstm|"
    r"transformer|self.supervised|few.shot|transfer learning|fine.tun|pretrain|"
    r"random forest|support vector|gradient boost|classification|detection|"
    r"recognition|embedding|encoder|spectrogram|mfcc|audio.classif|"
    r"sound.classif|sound.recogni|audio.recogni|feature extract)\b",
    re.IGNORECASE,
)

# Termos de áudio geral
AUDIO_RE = re.compile(
    r"\b(audio|sound|acoustic|spectrogram|mel.?spectrogram|mfcc|waveform|"
    r"bird|avian|parrot|psittac|vocali[sz]|bioacoustic|wildlife)\b",
    re.IGNORECASE,
)


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def is_arxiv(rec):
    # ArXiv IDs são numéricos tipo 2311.04945; OpenAlex começam com W
    return not rec.get("arxiv_id", "").startswith("W")


def keep_openalex(rec):
    cat = rec.get("primary_category", "")
    text = (rec.get("title", "") + " " + rec.get("abstract", ""))

    # Categorias explicitamente ML/áudio → sempre mantém
    if cat in OA_KEEP_CATS:
        return True

    # "Animal Vocal Communication" → inclui se tem ML OU análise acústica técnica
    if cat == "Animal Vocal Communication and Behavior":
        acoustic_analysis = re.compile(
            r"\b(bioacoustic|acoustic.analys|acoustic.classif|acoustic.monitor|"
            r"call.classif|call.recogni|vocali.classif|vocali.recogni|"
            r"spectrogram|mfcc|call.detect|sound.detect|acoustic.detect|"
            r"automatic.identif|automated.identif|machine.learning|deep.learning|"
            r"neural.network|classification|birdnet|birdclef|soundscape.monitor)\b",
            re.IGNORECASE,
        )
        return bool(acoustic_analysis.search(text))

    # Qualquer outra categoria: exige (birdnet/birdclef OU bioacoustic/passive acoustic monitor)
    # E ML presente — evita papers de conservação/genética sem IA
    strict_bio = re.compile(
        r"\b(bioacoustic|ecoacoustic|birdnet|birdclef|bird.?call.?recogni|"
        r"bird.?sound.?classif|acoustic.?species.?identif|passive.?acoustic.?monitor)\b",
        re.IGNORECASE,
    )
    if strict_bio.search(text) and ML_RE.search(text):
        return True

    return False


# --- Carrega ---
raw_arxiv  = load_jsonl("data/arxiv_raw.jsonl")
raw_oa     = load_jsonl("data/openalex_raw.jsonl")
raw_suppl  = load_jsonl("data/openalex_supplement.jsonl") if Path("data/openalex_supplement.jsonl").exists() else []
print(f"ArXiv raw: {len(raw_arxiv)} | OpenAlex raw: {len(raw_oa)} | Suplementar: {len(raw_suppl)}")

# --- Filtra ---
oa_kept   = [r for r in raw_oa    if keep_openalex(r)]
supl_kept = [r for r in raw_suppl if keep_openalex(r)]
print(f"OpenAlex após filtro: {len(oa_kept)} | Suplementar após filtro: {len(supl_kept)}")

# --- Mescla e normaliza ---
seen = set()
merged = []
for rec in raw_arxiv + oa_kept + supl_kept:
    doc_id = rec.get("arxiv_id", "")
    if doc_id and doc_id not in seen:
        seen.add(doc_id); merged.append(rec)

df = pd.DataFrame(merged)
df = df.drop_duplicates("arxiv_id", keep="last")
df = df[df["title"].str.len() > 0]
df = df[df["abstract"].str.len() > 50]

# Normaliza ano
def extract_year(val):
    import re as _re
    m = _re.search(r"(\d{4})", str(val))
    return m.group(1) if m else ""
df["published"] = df["published"].apply(extract_year)

print(f"Total final: {len(df)}")

cols = ["arxiv_id", "title", "abstract", "authors", "categories",
        "primary_category", "published", "doi"]
for c in cols:
    if c not in df.columns: df[c] = ""

with open("data/corpus.jsonl", "w", encoding="utf-8") as f:
    for _, row in df[cols].iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

print(f"Corpus salvo: data/corpus.jsonl ({len(df)} docs)")

# Stats
import collections, sys as _sys
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
cats = collections.Counter(df["primary_category"])
print("\nTop 12 categorias:")
for cat, n in cats.most_common(12):
    print(f"  {n:4d}  {cat}")

n_arxiv = sum(1 for _, r in df.iterrows() if not r["arxiv_id"].startswith("W"))
print(f"\nOrigem: ArXiv={n_arxiv} | OpenAlex={len(df)-n_arxiv}")
