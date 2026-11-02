import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class Retriever:
    """
    Hybrid Enterprise Retriever supporting:
    1. Semantic Similarity Search (via FAISS or Qdrant Vector Stores).
    2. QA-Aware Similarity Retrieval (matching queries against verified reference QA pairs).
    3. Metadata Filtering (by category, document title, source PDF, academic year).
    4. Neural Cross-Encoder Reranking for high-precision final candidate ranking.
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_manager: Any,
        reranker: Any,
        qa_index_mgr: Optional[Any] = None,
        collection_name: str = "enterprise_documents"
    ):
        self.vector_store = vector_store
        self.embedder = embedding_manager
        self.reranker = reranker
        self.qa_index_mgr = qa_index_mgr
        self.collection_name = collection_name

        # Compatibility alias if vector_store is Qdrant client
        self.qdrant = vector_store

    def _dense_search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executes dense vector search against FAISS or Qdrant backend."""
        query_embedding = self.embedder.embed_query(query)

        # 1. FAISS Backend
        if hasattr(self.vector_store, "search"):
            return self.vector_store.search(
                query_vector=query_embedding,
                top_k=k,
                filter_dict=filter_dict
            )

        # 2. Qdrant Backend Fallback
        q_vec = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        if hasattr(self.vector_store, "query_points"):
            results = self.vector_store.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                limit=k
            )
            points = results.points
        else:
            points = self.vector_store.search(
                collection_name=self.collection_name,
                query_vector=q_vec,
                limit=k
            )

        results = []
        for p in points:
            payload = p.payload or {}
            results.append({
                "id": p.id,
                "score": float(p.score) if p.score is not None else 0.0,
                "text": payload.get("text", ""),
                "raw_text": payload.get("raw_text", payload.get("text", "")),
                "source_file": payload.get("source_file", payload.get("source_pdf", "Unknown")),
                "source_pdf": payload.get("source_pdf", payload.get("source_file", "Unknown")),
                "page": payload.get("page", payload.get("page_number", 1)),
                "page_number": payload.get("page_number", payload.get("page", 1)),
                "document_title": payload.get("document_title", payload.get("source_file", "Document")),
                "category": payload.get("category", "General"),
                "academic_year": payload.get("academic_year", "-"),
                "program": payload.get("program", "-"),
                "payload": payload
            })
        return results

    def retrieve(
        self,
        query: str,
        retrieval_k: int = 8,
        final_k: int = 4
    ) -> List[Tuple[str, float]]:
        """
        Original legacy interface: retrieves top candidates and applies Cross-Encoder reranking.
        Returns list of (document_text, score) tuples for backward compatibility.
        """
        candidates = self._dense_search(query, k=retrieval_k)
        if not candidates:
            return []

        doc_texts = [c["text"] for c in candidates]
        reranked = self.reranker.rerank(query, doc_texts, top_k=final_k)
        return reranked

    def retrieve_hybrid(
        self,
        query: str,
        retrieval_k: int = 10,
        final_k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_qa_boost: bool = True
    ) -> Dict[str, Any]:
        """
        Complete Hybrid Retrieval:
        1. Dense Semantic Vector Search across document chunks.
        2. QA-Aware Similarity Search against verified Q&A index.
        3. Metadata Filtering (by category, document_title, source_pdf).
        4. Cross-Encoder Reranking of combined candidate chunks.
        """
        # 1. Semantic Search across document chunks
        primary_chunks = self._dense_search(query, k=retrieval_k, filter_dict=filter_dict)

        # 2. QA-Aware Question Similarity Search
        matched_qa = []
        qa_boosted_chunks = []
        if use_qa_boost and self.qa_index_mgr:
            matched_qa = self.qa_index_mgr.search_similar_questions(query, top_k=2)
            if matched_qa:
                top_match = matched_qa[0]
                target_doc = top_match.get("document")
                # If question specifically points to a document, retrieve top chunks from that document
                if target_doc:
                    doc_filter = {"source_pdf": target_doc}
                    qa_boosted_chunks = self._dense_search(
                        query=top_match["question"],
                        k=4,
                        filter_dict=doc_filter
                    )

        # 3. Merge & Deduplicate Candidates
        candidate_pool = []
        seen_chunk_ids = set()

        for chunk in primary_chunks + qa_boosted_chunks:
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            candidate_pool.append(chunk)

        if not candidate_pool:
            return {
                "query": query,
                "chunks": [],
                "matched_qa": matched_qa,
                "total_candidates": 0
            }

        # 4. Neural Cross-Encoder Reranking
        text_to_chunk = {c["text"]: c for c in candidate_pool}
        candidate_texts = list(text_to_chunk.keys())

        reranked_tuples = self.reranker.rerank(query, candidate_texts, top_k=final_k)

        final_chunks = []
        for rank, (doc_text, score) in enumerate(reranked_tuples, 1):
            original = text_to_chunk.get(doc_text, {})
            chunk_copy = dict(original)
            chunk_copy["rank"] = rank
            chunk_copy["rerank_score"] = float(score)
            chunk_copy["cosine_similarity"] = float(original.get("score", 0.0))
            chunk_copy["score"] = float(score)
            final_chunks.append(chunk_copy)

        return {
            "query": query,
            "chunks": final_chunks,
            "matched_qa": matched_qa,
            "total_candidates": len(candidate_pool)
        }