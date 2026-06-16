# Trabalho Prático — Inteligência Artificial (FACOM/UFMS, 2026/1)

**Aluno:** _Seu nome aqui_
**Matrícula:** _Sua matrícula_
**Nível:** Mestrado
**Tema do projeto de pesquisa:** Uma Revisão Experimental de Métodos de Inteligência Artificial para Reconhecimento e Verificação de Padrões de Vocalizações de Aves Psittacidae
**Tema da coleção (este trabalho):** Reconhecimento de vocalizações de aves com Inteligência Artificial (bioacústica) — ver nota de escopo abaixo

---

## Sobre o projeto

Este trabalho implementa o componente de **recuperação** ("R") de um sistema RAG aplicado à literatura científica sobre **reconhecimento de vocalizações de aves com Inteligência Artificial**. O sistema recebe uma consulta em texto livre e devolve uma lista ranqueada de artigos relevantes extraídos do ArXiv.

**Nota sobre o escopo:** o projeto de pesquisa do aluno tem como motivação específica as vocalizações de aves da família Psittacidae (papagaios, araras, periquitos, etc.). No entanto, a literatura científica indexada no ArXiv sobre Psittacidae especificamente é extremamente escassa. Por isso, o escopo da coleção foi ampliado para **bioacústica de aves em geral** (reconhecimento de canto/chamado de aves com deep learning), domínio no qual Psittacidae se insere e que é diretamente transferível para o projeto de pesquisa do aluno. Essa decisão de escopo está documentada na metodologia do relatório, conforme exigido pela Seção 3-(a) do enunciado.

A motivação é direta com o projeto de pesquisa: ao final do semestre, o sistema serve como um **retriever real para revisão bibliográfica** sobre aprendizado de máquina aplicado ao reconhecimento de vocalizações de aves.

---

## Estrutura do repositório

```
.
├── README.md                   <- este arquivo
├── requirements.txt            <- dependências Python
├── CHECKLIST.md                <- checklist de submissão
├── LINKS.txt                   <- link do vídeo e repositório
│
├── data/                       <- coleção (não versionar arquivos grandes no Git)
│   ├── arxiv_raw.jsonl         <- artigos brutos coletados da API
│   └── corpus.jsonl            <- corpus pré-processado
│
├── notebooks/
│   ├── 01_coleta_arxiv.ipynb   <- coleta via API do ArXiv
│   ├── 02_baseline_bm25.ipynb  <- recuperador BM25 (esparso)
│   ├── 03_retrieval_knn.ipynb  <- recuperador KNN/denso (TF-IDF ou embeddings)
│   ├── 04_modulo_aprofundamento.ipynb  <- módulo opcional (M1/M2/M3/M4/M5)
│   └── runs/                   <- arquivos .trec gerados pelos modelos
│       ├── bm25.trec
│       └── knn.trec
│
├── src/                        <- código Python reutilizável
│   ├── __init__.py
│   ├── preprocessing.py        <- tokenização, stopwords, normalização
│   ├── retrievers.py           <- classes BM25Retriever e KNNRetriever
│   └── utils.py                <- helpers (leitura JSONL, escrita TREC, etc.)
│
├── eval/
│   ├── queries.tsv             <- 10–20 queries de avaliação (criadas manualmente)
│   ├── qrels.tsv               <- relevance judgments anotados manualmente
│   └── evaluate.py             <- calcula P@k, R@k, MAP, nDCG (fornecido)
│
└── relatorio/
    ├── relatorio.tex           <- artigo formato SBC (LaTeX)
    └── relatorio.pdf           <- versão compilada para entrega
```

---

## Escopo da coleção

| Parâmetro            | Valor                                                                                      |
|----------------------|-------------------------------------------------------------------------------------------|
| **Tema**             | IA para reconhecimento de vocalizações de aves (bioacústica), com Psittacidae como motivação de pesquisa |
| **Palavras-chave**   | Ver lista completa abaixo                                                                  |
| **Categorias ArXiv** | usadas apenas para estatística/inspeção (`cs.SD`, `cs.LG`, `eess.AS`, `cs.CL`, `cs.CV`, `q-bio.*`) — não filtram a coleta |
| **Janela temporal**  | 2015–2026 (período de explosão do deep learning aplicado a bioacústica)                   |
| **Tamanho alvo**     | ~2.000 artigos                                                                            |
| **Tamanho final**    | _preencher após coleta_                                                                    |

### Palavras-chave para coleta (multi-query)

Para contornar o limite de resultados que o ArXiv aplica a queries compostas com muitos `OR`, a coleta é feita com **uma busca separada por keyword**, e os resultados são unidos e deduplicados por `arxiv_id`:

```python
KEYWORDS = [
    "bird sound",
    "bird call",
    "birdsong",
    "bioacoustic",
    "wildlife acoustic",
    "parrot",
]

YEAR_FROM = 2015
YEAR_TO   = 2026
```

> **Nota:** A query de coleta é ampla e visa trazer o máximo de artigos da área de bioacústica de aves. As queries de avaliação (Seção abaixo) são específicas e focadas — são duas coisas distintas.

---

## Queries de avaliação (10–20 queries de teste)

As queries simulam perguntas reais de um pesquisador da área. A maioria cobre bioacústica de aves em geral; algumas focam especificamente em papagaios/Psittacidae quando há literatura suficiente:

| ID  | Query                                                                 |
|-----|-----------------------------------------------------------------------|
| q01 | deep learning for parrot call recognition                             |
| q02 | convolutional neural network bird vocalization classification         |
| q03 | mel spectrogram feature extraction bird sound                        |
| q04 | transfer learning bioacoustics bird species identification           |
| q05 | self-supervised learning bird audio representation                   |
| q06 | attention mechanism bird call detection                               |
| q07 | dataset benchmark bird sound recognition                             |
| q08 | recurrent neural network bird song classification                    |
| q09 | parrot vocalization pattern analysis machine learning           |
| q10 | data augmentation audio bird species classification                  |
| q11 | few-shot learning bird sound recognition                             |
| q12 | transformer model audio classification wildlife                      |
| q13 | explainability interpretability bird call classifier                 |
| q14 | survey review deep learning bioacoustics                             |
| q15 | passive acoustic monitoring bird species                           |

> Estas queries serão usadas para gerar os pools de candidatos (top-10 de BM25 + top-10 de KNN), que serão anotados manualmente no `qrels.tsv`.

---

## Modelos implementados

| Modelo        | Tipo    | Biblioteca            | Arquivo de run   |
|---------------|---------|-----------------------|------------------|
| BM25          | Esparso | `rank_bm25`           | `runs/bm25.trec` |
| KNN + TF-IDF  | Denso   | `scikit-learn`        | `runs/knn.trec`  |
| _Módulo opt._ | —       | _a definir_           | `runs/modulo.trec` |

**Conexões com a disciplina:**
- BM25 → paradigma probabilístico (Naïve Bayes): ambos modelam relevância como probabilidade com independência entre termos
- KNN/denso → aula de KNN: em vez de votar rótulo, os K vizinhos mais próximos são devolvidos como ranking
- Módulo opcional → conforme escolha (regressão logística, clustering, regras de associação, etc.)

---

## Como reproduzir

```bash
# 1. Criar ambiente virtual e instalar dependências
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt

# 2. Coletar a coleção (ajuste keywords no notebook se necessário)
jupyter notebook notebooks/01_coleta_arxiv.ipynb

# 3. Rodar o baseline BM25
jupyter notebook notebooks/02_baseline_bm25.ipynb

# 4. Rodar o recuperador KNN
jupyter notebook notebooks/03_retrieval_knn.ipynb

# 5. Avaliar os dois modelos
python eval/evaluate.py \
    --qrels eval/qrels.tsv \
    --runs notebooks/runs/bm25.trec notebooks/runs/knn.trec \
    --k 10

# 6. Demo rápida (uma query de exemplo)
python src/demo.py "parrot call recognition convolutional neural network"
```

---

## Decisões de projeto

- **Tema/escopo:** Bioacústica de aves (reconhecimento de vocalizações com IA), com Psittacidae como motivação de pesquisa do aluno — literatura específica sobre Psittacidae no ArXiv é escassa, então o escopo foi ampliado conforme documentado na seção "Sobre o projeto".
- **Categorias do ArXiv:** usadas apenas para inspeção/estatística da coleção (predomínio observado: `cs.SD`, `cs.LG`, `eess.AS`).
- **Janela temporal:** 2015–2026 — captura o período de adoção de deep learning em bioacústica.
- **Pré-processamento:** lowercase, remoção de pontuação, remoção de stopwords em inglês (NLTK). Stemming avaliado experimentalmente.
- **Modelos implementados:** BM25 (`k1=1.5`, `b=0.75`), KNN com TF-IDF + similaridade do cosseno.
- **Módulo(s) de aprofundamento:** _a definir_

---

## Uso de assistentes de IA generativa

_Declaração a ser preenchida conforme o enunciado exige. Indicar onde e como foram utilizados assistentes de IA generativa (apoio à escrita, geração de trechos de código, sugestão de hiperparâmetros, etc.)._

---

## Vídeo de apresentação

URL: _https://..._

---

## Prazo de entrega

**23:59 do dia 19/06/2026** — AVA da disciplina (um único `.zip`).