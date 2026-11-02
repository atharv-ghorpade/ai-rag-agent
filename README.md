# Enterprise RAG — High-Precision Retrieval-Augmented Generation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-00599C?style=flat)](https://github.com/facebookresearch/faiss)
[![Cross-Encoder](https://img.shields.io/badge/Reranker-MS--MARCO--MiniLM-FF6F00?style=flat)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![Embeddings](https://img.shields.io/badge/Embedding-BGE--small--en--v1.5-blueviolet?style=flat)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, enterprise-grade **Retrieval-Augmented Generation (RAG)** platform designed for strict factual grounding across institutional records, engineering curricula, academic regulations, admissions policies, and departmental publications.

Featuring a modular **3-Persona Architecture**, **FAISS exact cosine vector indexing**, **QA-aware semantic boosting**, **two-stage cross-encoder neural reranking**, and **anti-hallucination structured synthesis** with verified source citations.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph P1["Person 1: Ingestion & Document Processing"]
        A["Institutional PDFs & Policies\n(Curriculum, Newsletters, Admissions)"] --> B["DocumentParser (PyMuPDF + OCR Engine)"]
        B --> C["Text Normalizer, Noise & Header Filter"]
        C --> D["RecursiveCharacterTextSplitter\n(700 chars, 100 overlap)"]
        D --> E["1,837 Context-Enriched Chunks\nwith Document & Page Metadata"]
    end

    subgraph P2["Person 2: FAISS Vector Store, QA Index & Reranking"]
        E --> F["Dense EmbeddingManager\n(BAAI/bge-small-en-v1.5, dim=384)"]
        F --> G[("FAISS Vector Database\n(IndexFlatIP, Normalized Cosine)")]
        QA["161 Verified Reference QA Pairs"] --> QAI[("QA Semantic Boosting Index\n(FAISS Vector Store)")]
        H["User Query"] --> I["Hybrid Stage 1 Search\n(Dense Vector + QA Boost + Metadata Filters)"]
        G --> I
        QAI --> I
        I --> J["Stage 2: Cross-Encoder Neural Reranker\n(ms-marco-MiniLM-L-6-v2)"]
        J --> K["Top-4 High-Confidence Chunks"]
    end

    subgraph P3["Person 3: Grounded Answer Synthesis & Citations"]
        K --> L["Strict Grounding Anti-Hallucination Prompt"]
        H --> L
        L --> M["Synthesizer Engine\n(Gemini / Groq / Ollama / Deterministic Extractive)"]
        M --> N["Structured Answer:\n## Answer\n## Details\n## Summary\n## Sources"]
    end
```

---

## Key Highlights

- **1,837 Indexed Context Chunks**: Ingested and chunked from 21 institutional documents, including the 87-page *B.Tech Computer Engineering Honours & Minor (R-2022) Curriculum*, quarterly *BYTES* departmental publications, merit lists, and admissions policies.
- **161 Ground-Truth Reference QA Pairs**: Curated question-answer pairs indexed in a dedicated FAISS semantic boosting store to ensure immediate sub-millisecond retrieval on common queries.
- **Two-Stage Re-Ranking Pipeline**:
  - **Stage 1 (Dense Vector)**: BAAI/bge-small-en-v1.5 embeddings with normalized inner product (`IndexFlatIP`) for fast retrieval.
  - **Stage 2 (Cross-Encoder)**: `ms-marco-MiniLM-L-6-v2` joint query-document neural attention scoring to eliminate vector space false positives.
- **Strict Grounding & Clean Formatting**:
  - Direct 1-sentence answer (`## Answer`).
  - Clean bulleted supporting explanations (`## Details`) without orphan list numbers or OCR artifacts.
  - Bulleted key points (`## Summary`).
  - Distinct vertical source citations (`## Sources`) with exact filename and page references.
- **Modern Glassmorphic Web Dashboard**: Pure Vanilla HTML5/CSS/JavaScript with zero bloated UI frameworks. Supports dark mode, real-time confidence scores, side-by-side Stage 1 vs Stage 2 comparison, and full corpus filtering.

---

## Directory Structure

```
ai-rag-agent/
├── data/
│   ├── new_docs/                        # Source PDFs (Curriculum, BYTES newsletters, policies)
│   ├── parsed_cache/                    # Extracted page text cache in JSON format
│   ├── faiss_index/                     # Persistent FAISS index (index.faiss & metadata.json)
│   ├── qa_index/                        # Dedicated FAISS index for QA semantic boosting
│   ├── qa_dataset.json                  # 161 verified reference QA pairs
│   ├── vit_chunks.json                  # 1,837 processed and contextualized text chunks
│   ├── sample_documents.json            # Baseline institutional policy chunks
│   └── evaluation_results.json          # Quantitative evaluation benchmark output
├── enterprise_rag/
│   ├── notebooks/
│   │   └── Person2_RAG_Project.ipynb    # Comprehensive development and evaluation notebook
│   └── src/
│       ├── __init__.py                  # Package exports
│       ├── config.py                    # Paths, model names, vector dimensions, chunk settings
│       ├── ingestion.py                 # Person 1: Document parser, chunker, text cleaning
│       ├── embeddings.py                # Person 2: Dense EmbeddingManager (BGE-small, 384-dim)
│       ├── faiss_manager.py             # Person 2: FAISS IndexFlatIP vector database manager
│       ├── qdrant_manager.py            # Person 2: Dual/fallback Qdrant vector store manager
│       ├── qa_retriever.py              # Person 2: QAIndexManager for semantic question boosting
│       ├── reranker.py                  # Person 2: Cross-Encoder neural reranker
│       ├── retriever.py                 # Person 2: Two-stage hybrid retriever
│       ├── evaluator.py                 # Person 2: Quantitative RAGEvaluator (Hit@K, MRR, F1)
│       ├── generator.py                 # Person 3: Anti-hallucination prompt and citation formatter
│       ├── pipeline.py                  # End-to-end Enterprise RAG orchestrator
│       ├── api.py                       # FastAPI REST service and static dashboard mount
│       └── web/                         # Production Web Interface
│           ├── index.html               # Semantic glassmorphic layout
│           ├── style.css                # Dark theme tokens, typography, and animations
│           └── app.js                   # Client-side asynchronous interaction logic
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                 # Automated unit tests covering FAISS, QA-search, and citations
├── .env.example                         # Environment configuration template
├── main.py                              # Master CLI command runner
├── requirements.txt                     # Project dependencies
└── README.md                            # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/atharv-ghorpade/ai-rag-agent.git
cd ai-rag-agent
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: If you wish to use cloud LLMs, add your `GEMINI_API_KEY` or `GROQ_API_KEY`. The system operates seamlessly out-of-the-box using local deterministic extractive synthesis).*

---

## Quickstart & CLI Commands

### 1. Run Pipeline Unit Tests
```bash
python -m unittest tests/test_pipeline.py
```
*Validates FAISS indexing, metadata filtering, QA semantic boosting, chunk deduplication, and citation generation.*

### 2. Ingest and Rebuild Vector Stores
```bash
python main.py ingest --use-cache
```
*Loads all parsed pages from `data/parsed_cache/`, applies `RecursiveCharacterTextSplitter`, embeds 1,837 chunks with BGE-small, and updates `data/faiss_index/`.*

### 3. Query the RAG Pipeline (Direct Q&A)
```bash
python main.py ask "What is the eligibility for Honours or Minor Degree in Computer Engineering?"
```
*Example Terminal Output:*
```text
QUESTION: "What is the eligibility for Honours or Minor Degree in Computer Engineering?"
[QA-Boost] Matched Reference Question: "What is the eligibility criteria for enrolling in an Honours or Minor degree program?" (sim: 0.941)

## Answer
Students must have a minimum CGPA of 6.75 up to the fourth semester without any active backlogs and at least 80 earned credits.

## Details
- Eligibility is open to students who maintain consistent academic performance through Semester IV.
- A total of 18 additional credits must be acquired across Semesters V through VIII.
- Students with standing ATKT or backlogs are ineligible to register.

## Summary
- Minimum CGPA required: 6.75.
- Required credits: 18 additional credits across 4 semesters.
- Active backlogs disqualify enrollment.

## Sources
- File: BTech_Computer_Engineering_Honours_Minor_R2022.pdf, Page: 5
```

### 4. Test Hybrid Retrieval with Metadata Filtering
```bash
python main.py hybrid "What is the total fee for MMS Institute Level admission?" --category "Admissions & Regulatory"
```

### 5. Run Quantitative Benchmark Evaluation
```bash
python main.py evaluate-qa
```

---

## Quantitative Evaluation & Benchmark Results

The pipeline was benchmarked against the 161 verified ground-truth QA dataset and multi-document query suite:

| Metric | Measured Score | Evaluation Rationale |
|---|---|---|
| **Hit@1 (Top-1 Accuracy)** | **95.65%** | Correct ground-truth document retrieved at absolute Rank 1 |
| **Hit@3 (Top-3 Recall)** | **97.39%** | Correct context present within the top 3 reranked chunks |
| **Hit@5 (Top-5 Recall)** | **97.39%** | Correct context present within top 5 candidates |
| **MRR (Mean Reciprocal Rank)** | **0.9652** | High-precision ordering across diverse query categories |
| **Context Precision@K** | **74.35%** | Proportion of retrieved chunks carrying direct factual context |
| **Context Recall@K** | **97.39%** | Completeness of relevant context passed to the generator |
| **Average Answer Correctness (F1)**| **73.12%** | Lexical and factual overlap against verified ground truth |
| **Average Cosine Similarity** | **0.6648** | Normalized vector similarity of retrieved chunks |
| **Average Retrieval Latency** | **~31.5 ms** | Sub-50ms dense vector retrieval via FAISS |
| **Average Total Pipeline Latency**| **~520 ms** | Dense retrieval + Cross-Encoder reranking |

---

## Web Dashboard & Interactive UI

Launch the production web dashboard:
```bash
python main.py serve --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

### Features:
1. **Q&A Intelligence**:
   - Natural language question input with quick-query recommendation chips.
   - Real-time confidence badges, latency readout, and structured markdown synthesis (`## Answer`, `## Details`, `## Summary`, `## Sources`).
   - Clickable verified source pills displaying document filename and exact page number.
2. **Vector vs Reranker Comparison**:
   - Interactive side-by-side view comparing **Stage 1 (FAISS Dense Cosine)** vs **Stage 2 (Cross-Encoder Neural Rerank)**.
   - Visual badges highlighting rank promotions, confidence scores, and document origin.
3. **Corpus Explorer**:
   - Searchable, real-time viewer browsing all **1,837 indexed document chunks**.
   - Instant client-side search filtering by keyword, academic year, and source document.

---

## REST API Documentation

The server exposes an asynchronous REST API documented interactively via OpenAPI at `http://127.0.0.1:8000/docs`:

| Method | Endpoint | Parameters | Description |
|---|---|---|---|
| `GET` | `/api/status` | — | Returns vector store type, chunk counts, QA pairs count, and model names |
| `POST` | `/api/ask` | `question`, `retrieval_k`, `final_k`, `use_qa_boost` | End-to-end grounded answer synthesis with verified citations |
| `POST` | `/api/query` | `query`, `retrieval_k`, `final_k`, `document`, `category` | Two-stage retrieval returning both Stage 1 and Stage 2 ranked chunks |
| `POST` | `/api/rerank` | `query`, `documents`, `top_k` | Cross-encoder reranker endpoint for arbitrary text arrays |
| `GET` | `/api/documents`| `limit`, `offset`, `search` | Corpus browser endpoint for exploring indexed chunks |

---

## Persona Responsibilities Breakdown

| Persona | Core Focus | Implementation Files |
|---|---|---|
| **Person 1** | **Ingestion, OCR & Chunking** | `ingestion.py`, `build_all_resources.py`, `data/new_docs/` |
| **Person 2** | **FAISS Indexing, QA Boosting & Reranking** | `embeddings.py`, `faiss_manager.py`, `qa_retriever.py`, `reranker.py`, `retriever.py`, `evaluator.py` |
| **Person 3** | **Anti-Hallucination Synthesis & Citations** | `generator.py`, `pipeline.py`, `api.py`, `enterprise_rag/src/web/` |

---

## License

This project is licensed under the [MIT License](LICENSE).
