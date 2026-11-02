import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .pipeline import EnterpriseRAGPipeline
from .evaluator import RAGEvaluator
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    COLLECTION_NAME,
    VECTOR_STORE_TYPE,
    VECTOR_SIZE,
    QA_DATASET_PATH
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(
    title="Enterprise RAG API",
    description="Enterprise RAG System featuring FAISS/Qdrant Vector Storage, QA-Aware Hybrid Retrieval, and Grounded Citations.",
    version="2.0.0"
)

# Enable CORS for local dev and frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline Singleton
_pipeline: Optional[EnterpriseRAGPipeline] = None


def get_pipeline() -> EnterpriseRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = EnterpriseRAGPipeline()
        # If vector store is empty, auto-ingest parsed cache
        if _pipeline.get_total_chunks() == 0:
            cache_dir = str(PROJECT_ROOT / "data" / "parsed_cache")
            if Path(cache_dir).exists():
                print(f"[API Startup] Indexing document cache from {cache_dir}...")
                _pipeline.ingest_from_parsed_cache(cache_dir=cache_dir)
    return _pipeline


# --- Pydantic Request/Response Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., example="Who is the Head of the Information Technology Department?")
    retrieval_k: int = Field(default=8, ge=1, le=50)
    final_k: int = Field(default=4, ge=1, le=20)
    category: Optional[str] = Field(default=None)
    document: Optional[str] = Field(default=None)


class AskRequest(BaseModel):
    question: str = Field(..., example="Who completed a PhD in Natural Language Processing?")
    retrieval_k: int = Field(default=10, ge=1, le=50)
    final_k: int = Field(default=4, ge=1, le=20)
    category: Optional[str] = Field(default=None)
    document: Optional[str] = Field(default=None)
    use_qa_boost: bool = Field(default=True)


class RerankRequest(BaseModel):
    query: str = Field(..., example="How many vacation days do employees get?")
    documents: List[str] = Field(..., example=[
        "Employees receive 20 days of vacation every year.",
        "The office is closed on national holidays.",
        "Workers are entitled to annual health insurance."
    ])
    top_k: int = Field(default=3, ge=1)


# --- REST API Endpoints ---

@app.get("/api/status")
def get_system_status():
    """Returns real-time health and metadata for the vector database, QA index, and ML models."""
    pipeline = get_pipeline()
    point_count = pipeline.get_total_chunks()
    qa_count = len(pipeline.qa_index_mgr.qa_pairs) if pipeline.qa_index_mgr else 0

    return {
        "status": "online",
        "system": "Enterprise RAG Intelligence System",
        "vector_store_type": pipeline.vector_store_type,
        "total_indexed_chunks": point_count,
        "qa_pairs_indexed": qa_count,
        "vector_dimension": pipeline.embedder.get_dimension(),
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "reranker_model": DEFAULT_RERANKER_MODEL,
        "llm_provider": os.getenv("LLM_PROVIDER", "auto/extractive")
    }


@app.post("/api/query")
def query_comparison(req: QueryRequest):
    """
    Executes dense vector search, applies cross-encoder reranking,
    and returns a side-by-side comparison with latency metrics.
    """
    t0 = time.perf_counter()
    pipeline = get_pipeline()
    filter_dict = {}
    if req.category:
        filter_dict["category"] = req.category
    if req.document:
        filter_dict["source_pdf"] = req.document

    res = pipeline.query_with_comparison(
        query_text=req.query,
        retrieval_k=req.retrieval_k,
        final_k=req.final_k,
        filter_dict=filter_dict if filter_dict else None
    )
    t_total = (time.perf_counter() - t0) * 1000.0

    return {
        "query": req.query,
        "latency_ms": round(t_total, 2),
        "vector_search_top_k": res["vector_search_top_k"],
        "reranked_top_k": res["reranked_top_k"]
    }


@app.post("/api/hybrid")
def hybrid_retrieval(req: QueryRequest):
    """
    Executes hybrid search combining dense semantic similarity,
    question similarity from verified QA dataset, and metadata filtering.
    """
    t0 = time.perf_counter()
    pipeline = get_pipeline()
    filter_dict = {}
    if req.category:
        filter_dict["category"] = req.category
    if req.document:
        filter_dict["source_pdf"] = req.document

    res = pipeline.hybrid_search(
        query=req.query,
        retrieval_k=req.retrieval_k,
        final_k=req.final_k,
        filter_dict=filter_dict if filter_dict else None,
        use_qa_boost=True
    )
    t_total = (time.perf_counter() - t0) * 1000.0

    return {
        "query": req.query,
        "latency_ms": round(t_total, 2),
        "total_candidates": res.get("total_candidates", 0),
        "matched_qa": res.get("matched_qa", []),
        "chunks": res.get("chunks", [])
    }


@app.post("/api/ask")
def ask_question(req: AskRequest):
    """
    End-to-end RAG question answering: retrieves chunks using hybrid search,
    reranks with cross-encoder, and generates a grounded answer with citations.
    """
    t0 = time.perf_counter()
    pipeline = get_pipeline()
    filter_dict = {}
    if req.category:
        filter_dict["category"] = req.category
    if req.document:
        filter_dict["source_pdf"] = req.document

    res = pipeline.ask(
        question=req.question,
        retrieval_k=req.retrieval_k,
        final_k=req.final_k,
        filter_dict=filter_dict if filter_dict else None,
        use_qa_boost=req.use_qa_boost
    )
    t_total = (time.perf_counter() - t0) * 1000.0

    return {
        "question": req.question,
        "answer": res["answer"],
        "citations": res["citations"],
        "matched_qa": res.get("matched_qa", []),
        "latency_ms": round(t_total, 2),
        "chunks_used": res.get("chunks_used", 0)
    }


@app.post("/api/rerank")
def rerank_texts(req: RerankRequest):
    """Direct cross-encoder reranking on arbitrary list of candidate texts."""
    pipeline = get_pipeline()
    ranked = pipeline.reranker.rerank(
        query=req.query,
        documents=req.documents,
        top_k=req.top_k
    )
    return {
        "query": req.query,
        "results": [
            {"rank": i, "document": doc, "score": float(score)}
            for i, (doc, score) in enumerate(ranked, 1)
        ]
    }


@app.get("/api/benchmark")
def run_benchmark(qa: bool = False, samples: Optional[int] = None):
    """Runs retrieval benchmark and returns quantitative metrics."""
    pipeline = get_pipeline()
    evaluator = RAGEvaluator(pipeline=pipeline)
    if qa:
        summary = evaluator.evaluate_dataset(max_samples=samples)
    else:
        summary = evaluator.run_benchmark()
    return summary


@app.get("/api/documents")
def list_documents(limit: int = 50, offset: int = 0, search: Optional[str] = None):
    """Browse indexed chunks from FAISS or Qdrant with search and pagination."""
    pipeline = get_pipeline()
    try:
        if pipeline.faiss_mgr:
            all_docs = pipeline.faiss_mgr.get_all_documents()
            if search:
                s_lower = search.lower()
                all_docs = [d for d in all_docs if s_lower in d.get("text", "").lower() or s_lower in d.get("source_pdf", "").lower()]
            paged = all_docs[offset:offset + limit]
            return {
                "total_fetched": len(paged),
                "total_available": len(all_docs),
                "documents": paged
            }
        elif pipeline.qdrant_mgr:
            points, _ = pipeline.qdrant_mgr.client.scroll(
                collection_name=pipeline.retriever.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            items = []
            for p in points:
                payload = p.payload or {}
                text = payload.get("text", "")
                if search and search.lower() not in text.lower():
                    continue
                items.append({
                    "id": p.id,
                    "text": text,
                    "source_file": payload.get("source_file", "Unknown"),
                    "page": payload.get("page", 1),
                    "academic_year": payload.get("academic_year", "-"),
                    "program": payload.get("program", "-")
                })
            return {
                "total_fetched": len(items),
                "documents": items
            }
        return {"total_fetched": 0, "documents": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Static Web UI Mounting ---

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
