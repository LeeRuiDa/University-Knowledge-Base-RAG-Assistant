# University Knowledge Base RAG Assistant

A retrieval-augmented assistant that answers questions over university policies and handbook documents with grounded source citations.

## Project snapshot

- Domain: `University of Nebraska-Lincoln` undergraduate CS policies and student rules
- Corpus: `24` official documents, `285` parsed sections, `383` indexed chunks
- Retrieval stack: `OpenRouter embeddings + Qdrant dense index + BM25 sparse index + reranking`
- Product surfaces: `React + TypeScript frontend + FastAPI API`, with Streamlit retained as a fallback

### Final metrics

| Evaluation | Metric | Score |
| --- | --- | ---: |
| Retrieval benchmark (59 questions) | Retrieval hit | `1.0000` |
| Retrieval benchmark (59 questions) | Top-1 hit | `0.9492` |
| Retrieval benchmark (59 questions) | Citation hit | `0.9831` |
| Answer-quality benchmark (18 questions) | Faithfulness mean | `0.9821` |
| Answer-quality benchmark (18 questions) | Completeness mean | `2.8571 / 3` |
| Answer-quality benchmark (18 questions) | Citation usefulness mean | `2.9444 / 3` |
| Answer-quality benchmark (18 questions) | Correct refusal rate | `1.0000` |

## Why this project

Generic chatbots often hallucinate or miss institution-specific rules. This project uses retrieval-augmented generation (RAG) to answer questions from curated university documents instead of relying on model memory alone.

It is designed to be a strong project because it demonstrates:

- document ingestion and parsing
- chunking and embeddings
- metadata-aware hybrid retrieval
- grounded answers with citations
- API and UI product surfaces
- retrieval evaluation and failure analysis on a real corpus
- answer-quality and refusal evaluation on a hosted-generation subset

## Current system

This repo currently implements a real-corpus RAG system with retrieval hardening:

- sync a public corpus from a manifest of official university URLs
- ingest PDF, HTML, Markdown, and text documents
- preserve `source`, `title`, `section`, `page`, `doc_type`, `year`, `url`, `program`, and `doc_id`
- chunk documents with a recursive splitter
- index dense vectors in Qdrant
- build a sparse BM25 index from chunk text
- fuse dense and sparse retrieval, rerank results, and diversify by source document
- answer questions with citations to retrieved chunks
- expose a typed FastAPI API, a production React frontend, and a Streamlit fallback
- compare dense vs hybrid retrieval against a handcrafted evaluation set
- evaluate hosted answers for faithfulness, completeness, citation usefulness, and refusal behavior

The current demo corpus is:

- University: `University of Nebraska-Lincoln`
- Scope: `Computer Science undergraduate program requirements + student academic policies`
- Source types: catalog pages, registrar pages, student accounts pages, financial aid pages, and official PDFs

## Sample queries

- `How many credits of CSCE 495 count as one tech elective course?`
- `When is priority registration for Fall Semester 2025?`
- `What late payment fee is assessed on delinquent student accounts?`
- `What is the deadline to appeal a parking ticket at UNL?`
  This should trigger a refusal because that policy is outside the indexed corpus.

## Architecture

```mermaid
flowchart LR
    A["Public UNL documents"] --> B["Manifest sync + parsing"]
    B --> C["Section-aware chunking"]
    C --> D["Embeddings + Qdrant"]
    C --> E["BM25 sparse index"]
    D --> F["Hybrid retrieval + reranking"]
    E --> F
    F --> G["Grounded answer + citations"]
    G --> H["FastAPI"]
    H --> I["React + TypeScript"]
    G --> J["Streamlit fallback"]
    K["Evaluation datasets"] --> L["Retrieval + answer evaluation"]
    L --> M["Reports and failure analysis"]
```

## Screenshots

### Answer with citations

![React answer with citations](reports/figures/react_answer_with_citations.png)

### Source panel

![React mobile evidence panel](reports/figures/react_mobile_evidence.png)

### Failure analysis artifact

![Answer evaluation artifact](reports/figures/answer_eval_failure_analysis.svg)

## Tech stack

- Python
- FastAPI
- React 19 + TypeScript + Vite
- React Markdown + sanitized Markdown rendering
- Vitest + React Testing Library
- Streamlit fallback
- LangChain text splitting and OpenAI-compatible integrations
- Qdrant
- rank-bm25
- pytest + ruff

## Local dev modes

This scaffold supports three modes:

1. Default local mode  
   Uses deterministic hash embeddings plus extractive answer synthesis so the app can run without paid APIs.
2. OpenAI mode  
   Uses OpenAI embeddings and generation.
3. OpenRouter mode  
   Uses OpenRouter's OpenAI-compatible API for embeddings and generation.

For a serious demo, use either OpenAI or OpenRouter mode. For smoke tests and offline development, local mode is enough, but retrieval quality will be lower than the hosted embedding-backed paths.

## Repo structure

```text
rag-assistant/
|- app/
|  |- fastapi_app.py
|  `- streamlit_app.py
|- frontend/
|  |- src/
|  |  |- api/
|  |  |- components/
|  |  |- hooks/
|  |  |- styles/
|  |  `- types/
|  |- package.json
|  `- vite.config.ts
|- data/
|  |- corpus_manifest.csv
|  |- raw/
|  |- parsed/
|  `- eval/
|- reports/
|  `- figures/
|- src/
|  |- answer_eval.py
|  |- answer.py
|  |- chunking.py
|  |- config.py
|  |- corpus.py
|  |- embed.py
|  |- evaluate.py
|  |- ingest.py
|  |- loaders.py
|  |- models.py
|  |- predict.py
|  |- retriever.py
|  `- sparse_index.py
|- tests/
|- Dockerfile
|- requirements.txt
`- README.md
```

## Getting started

### 1. Install backend dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional evaluation dependencies:

```powershell
pip install -r requirements-eval.txt
```

### 2. Configure the environment

```powershell
Copy-Item .env.example .env
```

For local React development, the default CORS origins in `.env.example` are sufficient. For a hosted frontend, replace them with the exact public origin; wildcard origins are rejected.

For OpenAI:

```env
OPENAI_API_KEY=your_key_here
EMBEDDING_PROVIDER=openai
GENERATION_PROVIDER=openai
```

For OpenRouter:

```env
OPENROUTER_API_KEY=your_key_here
EMBEDDING_PROVIDER=openrouter
GENERATION_PROVIDER=openrouter
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_CHAT_MODEL=openai/gpt-4.1-mini
```

### 3. Sync and ingest the corpus

The repo uses `data/corpus_manifest.csv` as the source of truth for public document URLs and metadata. During ingestion, the app downloads those files into `data/raw/unl/` and indexes them.

```powershell
python -m src.ingest --recreate
```

### 4. Run the API

```powershell
uvicorn app.fastapi_app:app --reload
```

### 5. Run the React frontend

The frontend requires Node.js `20.19` or newer. In a second terminal:

```powershell
Set-Location frontend
Copy-Item .env.example .env
npm install
npm run dev
```

`VITE_API_BASE_URL` is the only browser environment variable. It must point to the FastAPI base URL and must not contain credentials.

### 6. Run the Streamlit fallback

```powershell
streamlit run app/streamlit_app.py
```

## API endpoints

- Public frontend endpoints: `GET /health`, `GET /metadata`, and `POST /ask`
- Administrative endpoints: `POST /ingest` and `POST /reindex`

Administrative endpoints require `X-Admin-Key` to match `ADMIN_API_KEY`. If `ADMIN_API_KEY` is empty, both endpoints return `503` and remain disabled. Keep these endpoints protected or disabled whenever the backend is public.

## Environment variables

| Variable | Runtime | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | Browser build | Public FastAPI base URL; never include secrets |
| `CORS_ALLOWED_ORIGINS` | FastAPI | Comma-separated exact frontend origins |
| `ADMIN_API_KEY` | FastAPI | Enables protected ingestion and reindexing operations |
| `QDRANT_URL` / `QDRANT_API_KEY` | FastAPI | Remote vector database connection for hosted deployments |
| `EMBEDDING_PROVIDER` / `GENERATION_PROVIDER` | FastAPI | Selects local, OpenAI, or OpenRouter providers |
| `OPENAI_API_KEY` / `OPENROUTER_API_KEY` | FastAPI only | Hosted model credentials; never expose to Vite |

See `.env.example` and `frontend/.env.example` for the complete development configuration.

## Production build and deployment

Build the static frontend:

```powershell
Set-Location frontend
npm install
npm run lint
npm run test
npm run build
```

Deploy `frontend/dist/` to a static host and deploy the repository `Dockerfile` as the separate FastAPI service. Configure `VITE_API_BASE_URL` at frontend build time and set the deployed frontend origin in `CORS_ALLOWED_ORIGINS` on the backend. No SPA rewrite rule is required because the application has a single route.

Use a remote Qdrant service for the hosted demo. Local Qdrant storage is intended for single-process development and cannot safely support independent hosted API instances. This repository does not claim a live deployment URL.

## Retrieval evaluation

The repo includes a real evaluation set at `data/eval/unl_cs_policies_eval.csv`.

Run a single evaluation:

```powershell
python -m src.evaluate
```

Run the dense vs hybrid comparison:

```powershell
python -m src.evaluate --compare --generation-provider extractive
```

This writes:

- `reports/eval_results.json`
- `reports/retrieval_comparison.json`
- `reports/retrieval_failure_analysis.md`

The reports include:

- expected-document retrieval hit rate
- top-1 retrieval hit rate
- citation hit rate
- category breakdowns
- example failures for manual debugging
- dense vs hybrid fixes, remaining failures, and regressions

Current benchmark on the 59-question UNL eval set:

- Dense: retrieval `0.9831`, top-1 `0.7797`, citation `0.9322`
- Hybrid: retrieval `1.0000`, top-1 `0.9492`, citation `0.9831`

## Answer-quality evaluation

The repo also includes a curated hosted-generation subset at `data/eval/unl_cs_answer_eval_subset.csv`.

Run the answer-quality evaluation:

```powershell
python -m src.answer_eval --generation-provider openrouter --judge-provider openrouter
```

This writes:

- `reports/answer_eval_results.json`
- `reports/answer_failure_analysis.md`

The answer-quality run stores the generated answer, retrieved chunks, cited sources, refusal behavior, and config used for each question. It scores:

- Ragas faithfulness on supported questions
- a custom 0-3 completeness rubric against reference answers
- citation usefulness with deterministic checks against cited chunks
- refusal behavior on out-of-corpus questions

Current benchmark on the 18-question hosted-generation subset:

- Faithfulness mean: `0.9821`
- Completeness mean: `2.8571`
- Citation usefulness mean: `2.9444`
- Refusal behavior mean: `3.0`
- Supported pass rate: `0.8571`
- Correct refusal rate: `1.0`

The ingestion step also writes chunk inspection artifacts under `data/parsed/`, including:

- `chunk_preview.jsonl`
- `chunk_catalog.jsonl`
- `chunk_stats.json`
- `ingestion_manifest.json`

## Notes

- This is an independent portfolio project using public University of Nebraska-Lincoln documents. It is not an official UNL product.
- The current corpus uses public official UNL pages and documents captured through `data/corpus_manifest.csv`.
- `data/raw/` and `data/parsed/` are generated during ingestion and are intentionally not committed.
- `reports/*.json` are generated benchmark outputs and are intentionally not committed.
- Binary PDFs are re-downloaded from the manifest during ingestion instead of being vendored in the repo.
- OpenRouter can be used through its OpenAI-compatible endpoint and embeddings API: [Quickstart](https://openrouter.ai/docs/quickstart) and [Embeddings](https://openrouter.ai/docs/api-reference/embeddings).
- Qdrant local mode is convenient for development, but it does not support concurrent access from multiple Python processes. For shared API/UI access or parallel evaluation, set `QDRANT_URL` and use a normal Qdrant server.
