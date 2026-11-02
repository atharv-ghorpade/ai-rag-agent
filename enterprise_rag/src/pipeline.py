import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from .config import (
    ROOT_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    COLLECTION_NAME,
    VECTOR_SIZE,
    VECTOR_STORE_TYPE,
    FAISS_STORAGE_PATH,
    QA_DATASET_PATH
)
from .embeddings import EmbeddingManager
from .reranker import Reranker
from .retriever import Retriever
from .faiss_manager import FAISSManager
from .qa_retriever import QAIndexManager
from .generator import RAGGenerator


class EnterpriseRAGPipeline:
    """
    Complete end-to-end Enterprise RAG Pipeline:
    - Modular Vector Storage: FAISS (default high-performance) and Qdrant (persistent cloud/embedded).
    - Dense SentenceTransformer embeddings (BAAI/bge-small-en-v1.5).
    - Neural Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2).
    - QA-Aware Semantic Indexing over verified reference question-answer pairs.
    - Hybrid Retrieval (semantic search + metadata filtering + QA similarity).
    - Grounded Answer Generation with precise source and page citations.
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
        vector_store_type: str = VECTOR_STORE_TYPE,
        faiss_storage_path: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
        qdrant_client: Optional[Any] = None,
        load_qa_index: bool = True
    ):
        print("=" * 60)
        print(f" Initializing Enterprise RAG Pipeline (Store: {vector_store_type.upper()})")
        print("=" * 60)

        self.vector_store_type = vector_store_type.lower()

        # 1. Initialize Embedder
        self.embedder = EmbeddingManager(model_name=embedding_model_name)
        dim = self.embedder.get_dimension()

        # 2. Initialize Vector Database Backend
        if self.vector_store_type == "faiss":
            self.faiss_mgr = FAISSManager(
                dimension=dim,
                storage_path=faiss_storage_path or FAISS_STORAGE_PATH
            )
            self.vector_store = self.faiss_mgr
            self.qdrant_mgr = None
        else:
            from .qdrant_manager import QdrantManager, create_qdrant_client
            client = qdrant_client or create_qdrant_client()
            self.qdrant_mgr = QdrantManager(
                client=client,
                collection_name=collection_name,
                vector_size=dim
            )
            self.vector_store = self.qdrant_mgr.client
            self.faiss_mgr = None

        # 3. Initialize QA Index Manager
        self.qa_index_mgr = None
        if load_qa_index:
            try:
                self.qa_index_mgr = QAIndexManager(
                    embedding_manager=self.embedder,
                    qa_dataset_path=QA_DATASET_PATH
                )
            except Exception as e:
                print(f"[Notice] QA Index could not be loaded: {e}")

        # 4. Initialize Reranker
        self.reranker = Reranker(model_name=reranker_model_name)

        # 5. Initialize Hybrid Retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedder,
            reranker=self.reranker,
            qa_index_mgr=self.qa_index_mgr,
            collection_name=collection_name
        )

        # 6. Initialize Generator
        self.generator = RAGGenerator()

        print("=" * 60)
        print(f" Enterprise RAG Pipeline ready! (Indexed chunks: {self.get_total_chunks()})")
        print("=" * 60)

    def get_total_chunks(self) -> int:
        """Returns the total number of documents indexed in the vector store."""
        if self.faiss_mgr:
            return self.faiss_mgr.count()
        elif self.qdrant_mgr:
            return self.qdrant_mgr.count()
        return 0

    def index_documents(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[Any]] = None
    ) -> int:
        """
        Embeds documents and upserts them into the active vector store (FAISS or Qdrant).
        """
        print(f"\n[Indexing] Embedding {len(documents)} document(s)...")
        embeddings = self.embedder.embed_documents(documents)

        if self.faiss_mgr:
            count = self.faiss_mgr.add_documents(
                texts=documents,
                embeddings=embeddings,
                metadata_list=metadata_list,
                ids=ids
            )
            print(f"[Indexing: FAISS] Added {count} documents (Total indexed: {self.faiss_mgr.count()})")
            return count
        else:
            count = self.qdrant_mgr.upsert_documents(
                texts=documents,
                embeddings=embeddings,
                metadata_list=metadata_list,
                ids=ids
            )
            print(f"[Indexing: Qdrant] Added {count} documents (Total indexed: {self.qdrant_mgr.count()})")
            return count

    def index_from_json(self, json_filepath: str) -> int:
        """
        Loads document chunks from a JSON file and indexes them into the vector database.
        """
        path = Path(json_filepath)
        if not path.is_absolute():
            path = ROOT_DIR / json_filepath

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        metadata_list = []
        ids = []

        for item in data:
            if isinstance(item, str):
                texts.append(item)
                metadata_list.append({})
            elif isinstance(item, dict):
                text = item.get("text") or item.get("raw_text", "")
                texts.append(text)
                meta = {k: v for k, v in item.items() if k not in ("text", "id")}
                metadata_list.append(meta)
                if "id" in item or "chunk_id" in item:
                    ids.append(item.get("id") or item.get("chunk_id"))

        # Batch indexing in chunks of 32 for memory efficiency
        batch_size = 32
        total_added = 0
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i + batch_size]
            b_meta = metadata_list[i:i + batch_size]
            b_ids = ids[i:i + batch_size] if ids else None
            total_added += self.index_documents(b_texts, b_meta, b_ids)

        return total_added

    def ingest_from_parsed_cache(
        self,
        cache_dir: Optional[str] = None,
        output_json: Optional[str] = None
    ) -> int:
        """
        Ingests all documents from data/parsed_cache/ using RecursiveCharacterTextSplitter
        and indexes every chunk into the vector database.
        """
        from .ingestion import IngestionPipeline

        ingestion = IngestionPipeline()
        chunks = ingestion.ingest_from_parsed_cache(
            cache_dir=cache_dir,
            output_json_path=output_json or str(ROOT_DIR / "data" / "vit_chunks.json")
        )

        texts = [c["text"] for c in chunks]
        metadata_list = [
            {k: v for k, v in c.items() if k not in ("text", "id")}
            for c in chunks
        ]
        ids = [c["chunk_id"] for c in chunks]

        # Index in batches of 32
        batch_size = 32
        total_indexed = 0
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i + batch_size]
            b_meta = metadata_list[i:i + batch_size]
            b_ids = ids[i:i + batch_size]
            total_indexed += self.index_documents(b_texts, b_meta, b_ids)
            print(f"  Indexed [{total_indexed}/{len(texts)}] chunks...")

        return total_indexed

    def ingest_directory(
        self,
        directory_path: str,
        output_json: Optional[str] = None
    ) -> int:
        """
        Parses all PDFs in a directory with OCR fallback and indexes into vector store.
        """
        from .ingestion import IngestionPipeline

        ingestion = IngestionPipeline()
        chunks = ingestion.ingest_directory(directory_path, output_json_path=output_json)

        texts = [c["text"] for c in chunks]
        metadata_list = [
            {k: v for k, v in c.items() if k not in ("text", "id")}
            for c in chunks
        ]
        ids = [c["chunk_id"] for c in chunks]

        batch_size = 32
        total_indexed = 0
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i + batch_size]
            b_meta = metadata_list[i:i + batch_size]
            b_ids = ids[i:i + batch_size]
            total_indexed += self.index_documents(b_texts, b_meta, b_ids)

        return total_indexed

    def query(
        self,
        query_text: str,
        retrieval_k: int = 8,
        final_k: int = 4
    ) -> List[Tuple[str, float]]:
        """Standard legacy query interface returning (doc_text, score) tuples."""
        return self.retriever.retrieve(query=query_text, retrieval_k=retrieval_k, final_k=final_k)

    def query_with_comparison(
        self,
        query_text: str,
        retrieval_k: int = 8,
        final_k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes query and returns comparison between initial dense vector search vs reranked results.
        """
        dense_results = self.retriever._dense_search(query_text, k=retrieval_k, filter_dict=filter_dict)
        candidate_texts = [item["text"] for item in dense_results]
        reranked = self.reranker.rerank(query_text, candidate_texts, top_k=final_k)

        text_to_item = {item["text"]: item for item in dense_results}
        reranked_detailed = []
        for rank, (doc_text, score) in enumerate(reranked, 1):
            original = text_to_item.get(doc_text, {})
            reranked_detailed.append({
                "rank": rank,
                "score": float(score),
                "text": doc_text,
                "raw_text": original.get("raw_text", doc_text),
                "source_file": original.get("source_file") or original.get("source_pdf", "Unknown"),
                "source_pdf": original.get("source_pdf") or original.get("source_file", "Unknown"),
                "page": original.get("page") or original.get("page_number", 1),
                "page_number": original.get("page_number") or original.get("page", 1),
                "document_title": original.get("document_title", "Document"),
                "category": original.get("category", "General"),
                "payload": original.get("payload", {})
            })

        return {
            "query": query_text,
            "vector_search_top_k": dense_results,
            "reranked_top_k": reranked_detailed
        }

    def hybrid_search(
        self,
        query: str,
        retrieval_k: int = 10,
        final_k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_qa_boost: bool = True
    ) -> Dict[str, Any]:
        """
        Executes hybrid search combining:
        1. Dense semantic vector similarity (FAISS/Qdrant)
        2. QA-Aware question similarity (via QAIndexManager)
        3. Metadata filtering (by category, document title, etc.)
        4. Cross-Encoder neural reranking
        """
        return self.retriever.retrieve_hybrid(
            query=query,
            retrieval_k=retrieval_k,
            final_k=final_k,
            filter_dict=filter_dict,
            use_qa_boost=use_qa_boost
        )

    def ask(
        self,
        question: str,
        retrieval_k: int = 10,
        final_k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_qa_boost: bool = True
    ) -> Dict[str, Any]:
        """
        Complete end-to-end question answering pipeline:
        Retrieves top candidate chunks using hybrid search, reranks with cross-encoder,
        and generates an anti-hallucinatory answer with exact source and page citations.
        """
        hybrid_res = self.hybrid_search(
            query=question,
            retrieval_k=retrieval_k,
            final_k=final_k,
            filter_dict=filter_dict,
            use_qa_boost=use_qa_boost
        )

        top_chunks = hybrid_res["chunks"]
        matched_qa = hybrid_res.get("matched_qa", [])

        gen_res = self.generator.generate_answer(
            query=question,
            retrieved_chunks=top_chunks,
            matched_qa=matched_qa
        )

        return {
            "query": question,
            "question": question,
            "answer": gen_res["answer"],
            "citations": gen_res["citations"],
            "matched_qa": matched_qa,
            "chunks_used": len(top_chunks),
            "retrieved_chunks": top_chunks
        }
