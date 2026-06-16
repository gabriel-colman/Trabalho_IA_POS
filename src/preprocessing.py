"""Pré-processamento de texto para recuperação de informação."""
import re

import nltk

try:
    from nltk.corpus import stopwords
    _STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    _STOP = set(stopwords.words("english"))

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def tokenize(text: str, use_stopwords: bool = True) -> list[str]:
    """
    Tokenização básica: lowercase → extrai tokens alfabéticos ≥ 3 chars
    → remove stopwords inglesas.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if use_stopwords:
        tokens = [t for t in tokens if t not in _STOP and len(t) > 2]
    else:
        tokens = [t for t in tokens if len(t) > 2]
    return tokens


def doc_text(doc: dict) -> str:
    """Concatena título + abstract para indexação."""
    return (doc.get("title", "") + ". " + doc.get("abstract", "")).strip()


def tokenize_corpus(docs: list[dict]) -> list[list[str]]:
    """Tokeniza lista de documentos (título + abstract)."""
    return [tokenize(doc_text(d)) for d in docs]
