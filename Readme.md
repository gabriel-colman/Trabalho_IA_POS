# Trabalho Prático - Inteligência Artificial (FACOM/UFMS, 2026/1)

**Aluno:** Gabriel Colman  
**Nível:** Mestrado  
**Tema do projeto de pesquisa:** Uma Revisão Experimental de Métodos de Inteligência Artificial para Reconhecimento e Verificação de Padrões de Vocalizações de Aves Psittacidae  
**Tema da coleção:** Reconhecimento de vocalizações de aves com Inteligência Artificial (bioacústica)

---

## Sobre o Projeto

Este projeto implementa o componente de **recuperação** ("R") de um sistema RAG aplicado à literatura científica sobre **reconhecimento de vocalizações de aves com Inteligência Artificial**.

O sistema recebe consultas em texto livre e devolve rankings de artigos científicos relacionados ao tema. A motivação original do projeto está nas vocalizações de aves da família **Psittacidae** (papagaios, araras, periquitos etc.). Como a literatura específica sobre Psittacidae no ArXiv é limitada, a coleção foi ampliada para bioacústica de aves em geral, mantendo relação direta com o tema de pesquisa.

---

## Estrutura do Projeto

```text
.
├── Readme.md
├── requirements.txt
├── Trabalho_2026-1.pdf
├── data/
│   ├── arxiv_raw.jsonl
│   ├── openalex_raw.jsonl
│   ├── openalex_supplement.jsonl
│   └── corpus.jsonl
├── eval/
│   ├── queries.tsv
│   ├── qrels.tsv
│   └── evaluate.py
├── notebooks/
│   ├── 01_coleta_arxiv.ipynb
│   ├── 02_baseline_bm25.ipynb
│   ├── 03_retrieval_knn.ipynb
│   ├── 04_modulo_aprofundamento.ipynb
│   └── runs/
│       ├── bm25.trec
│       ├── knn_tfidf.trec
│       └── hybrid_rrf.trec
├── scripts/
│   ├── collect_supplement.py
│   ├── rebuild_corpus.py
│   └── gen_runs.py
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── retrievers.py
    └── utils.py
```

---

## Corpus

| Item | Valor |
|---|---|
| Tema | IA para reconhecimento de vocalizações de aves |
| Escopo | Bioacústica de aves, com Psittacidae como motivação |
| Fontes | ArXiv e OpenAlex |
| Janela temporal | 2015-2026 |
| Tamanho final | 1092 documentos |
| Formato | JSONL |
| Arquivo principal | `data/corpus.jsonl` |

Cada documento do corpus possui, quando disponível:

- `arxiv_id`
- `title`
- `abstract`
- `authors`
- `categories`
- `primary_category`
- `published`
- `doi`

---

## Modelos Implementados

| Modelo | Tipo | Implementação | Saída |
|---|---|---|---|
| BM25 | Recuperação esparsa | `rank_bm25` | `notebooks/runs/bm25.trec` |
| TF-IDF + KNN | Recuperação vetorial | `scikit-learn` + cosseno | `notebooks/runs/knn_tfidf.trec` |
| Híbrido RRF | Fusão de rankings | BM25 + TF-IDF por Reciprocal Rank Fusion | `notebooks/runs/hybrid_rrf.trec` |

O módulo de aprofundamento escolhido foi o **M5 - Ranking híbrido sparse+dense**, combinando BM25 e TF-IDF/KNN por **Reciprocal Rank Fusion (RRF)**.

---

## Como Executar Pelo Terminal

Os comandos abaixo assumem que você está na raiz do projeto.

### 1. Clonar o repositório

```powershell
git clone https://github.com/gabriel-colman/Trabalho_IA_POS.git
cd Trabalho_IA_POS
```

### 2. Criar e ativar o ambiente virtual

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação do ambiente, execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para abrir e executar os notebooks, instale também:

```powershell
pip install notebook
```

### 4. Verificar se o código Python compila

```powershell
python -m compileall src eval scripts
```

### 5. Gerar os arquivos de ranking

Este comando usa o corpus em `data/corpus.jsonl`, lê as consultas em `eval/queries.tsv` e gera os arquivos `.trec` dentro de `notebooks/runs/`.

```powershell
python scripts/gen_runs.py
```

Arquivos esperados:

```text
notebooks/runs/bm25.trec
notebooks/runs/knn_tfidf.trec
notebooks/runs/hybrid_rrf.trec
```

### 6. Avaliar os modelos

```powershell
python eval/evaluate.py `
  --qrels eval/qrels.tsv `
  --runs notebooks/runs/bm25.trec notebooks/runs/knn_tfidf.trec notebooks/runs/hybrid_rrf.trec `
  --k 10
```

No Linux ou macOS:

```bash
python eval/evaluate.py \
  --qrels eval/qrels.tsv \
  --runs notebooks/runs/bm25.trec notebooks/runs/knn_tfidf.trec notebooks/runs/hybrid_rrf.trec \
  --k 10
```

O script imprime, para cada modelo:

- `P@10`
- `R@10`
- `AP`
- `nDCG@10`
- média geral por sistema

### 7. Executar os notebooks

```powershell
jupyter notebook
```

Ordem sugerida:

1. `notebooks/01_coleta_arxiv.ipynb`
2. `notebooks/02_baseline_bm25.ipynb`
3. `notebooks/03_retrieval_knn.ipynb`
4. `notebooks/04_modulo_aprofundamento.ipynb`

---

## Recriar o Corpus

O projeto já contém o corpus consolidado em `data/corpus.jsonl`. Caso seja necessário reconstruí-lo a partir dos arquivos brutos:

```powershell
python scripts/rebuild_corpus.py
python scripts/gen_runs.py
```

Para executar a coleta suplementar via OpenAlex:

```powershell
python scripts/collect_supplement.py
python scripts/rebuild_corpus.py
python scripts/gen_runs.py
```

---

## Avaliação

As consultas de avaliação estão em:

```text
eval/queries.tsv
```

Os julgamentos de relevância estão em:

```text
eval/qrels.tsv
```

Formato do `qrels.tsv`:

```text
qid 0 doc_id relevancia
```

Onde:

- `0` = não relevante
- `1` = relevante
- `2` = muito relevante

### Resultados atuais

Avaliação executada com `k=10` sobre 15 consultas e 218 julgamentos de relevância.

| Sistema | P@10 | R@10 | AP | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.4467 | 0.4404 | 0.4899 | 0.5203 |
| TF-IDF + KNN | 0.5133 | 0.5110 | 0.6156 | 0.6379 |
| Híbrido RRF | **0.5467** | **0.5340** | **0.6236** | **0.6608** |

O melhor desempenho médio foi obtido pelo **Híbrido RRF**, indicando que a fusão entre o ranking esparso do BM25 e o ranking vetorial por TF-IDF/KNN recupera documentos relevantes de forma mais equilibrada do que cada abordagem isolada.

---

## Observações Importantes

- A pasta `.venv/` não deve ser enviada ao GitHub.
- Arquivos de cache Python (`__pycache__/`) e checkpoints de notebooks também são ignorados.
- Os rankings são salvos no formato TREC.
- A avaliação depende da qualidade dos julgamentos em `eval/qrels.tsv`; os julgamentos atuais são uma anotação inicial assistida e devem ser revisados antes da entrega final.

---

## Uso de IA Generativa

Assistentes de IA generativa foram utilizados como apoio para organização do projeto, revisão textual, sugestões de estrutura de código, documentação e identificação de pontos pendentes. A implementação e validação final permanecem sob responsabilidade do autor.
