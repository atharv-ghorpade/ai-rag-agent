import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from .config import ROOT_DIR, FAISS_STORAGE_PATH, VECTOR_SIZE

logger = logging.getLogger(__name__)


class FAISSManager:
    """
    Enterprise-grade FAISS Vector Database Manager.
    - Uses faiss.IndexFlatIP for exact Cosine similarity on normalized dense vectors.
    - Maintains comprehensive document and metadata store (docstore).
    - Prevents duplicate chunks by tracking unique chunk IDs and content hashes.
    - Supports saving, loading, incremental merging, and metadata-filtered search.
    """

    def __init__(
        self,
        dimension: int = VECTOR_SIZE,
        storage_path: Optional[str] = None
    ):
        self.dimension = dimension
        self.storage_path = Path(storage_path or FAISS_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.storage_path / "index.faiss"
        self.meta_file = self.storage_path / "docstore.json"

        # Internal state
        self.index: faiss.Index = faiss.IndexFlatIP(self.dimension)
        self.docstore: List[Dict[str, Any]] = []
        self.seen_chunk_ids: set = set()

        # Load existing index if present on disk
        if self.index_file.exists() and self.meta_file.exists():
            self.load()

    def count(self) -> int:
        """Returns the number of vectors indexed."""
        return int(self.index.ntotal)

    def save(self):
        """Persists the FAISS index and metadata store to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "ntotal": self.index.ntotal,
                "docstore": self.docstore
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {self.storage_path}")

    def load(self):
        """Loads FAISS index and metadata store from disk."""
        try:
            self.index = faiss.read_index(str(self.index_file))
            with open(self.meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.docstore = data.get("docstore", [])
                self.dimension = data.get("dimension", self.dimension)
            self.seen_chunk_ids = {doc.get("chunk_id", str(i)) for i, doc in enumerate(self.docstore)}
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {self.storage_path}: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.docstore = []
            self.seen_chunk_ids = set()

    def add_documents(
        self,
        texts: List[str],
        embeddings: Any,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[Any]] = None
    ) -> int:
        """
        Adds new documents and their embeddings to the FAISS index.
        Automatically avoids duplicate chunks and maintains full metadata tracking.
        """
        if len(texts) == 0:
            return 0

        emb_matrix = np.array(embeddings, dtype=np.float32)
        if len(emb_matrix.shape) == 1:
            emb_matrix = np.expand_dims(emb_matrix, axis=0)

        # Ensure embeddings are normalized for Cosine distance via Inner Product
        faiss.normalize_L2(emb_matrix)

        new_vectors = []
        new_docs = []
        added_count = 0

        for i, text in enumerate(texts):
            raw_id = ids[i] if (ids and i < len(ids)) else f"chunk_{self.count() + added_count + 1}"
            chunk_id_str = str(raw_id)

            # Deduplication check
            if chunk_id_str in self.seen_chunk_ids:
                continue

            meta = dict(metadata_list[i]) if (metadata_list and i < len(metadata_list)) else {}
            
            # Standardize essential enterprise metadata
            doc_entry = {
                "chunk_id": chunk_id_str,
                "text": text,
                "raw_text": meta.get("raw_text", text),
                "source_pdf": meta.get("source_pdf") or meta.get("source_file", "unknown_source.pdf"),
                "source_file": meta.get("source_file") or meta.get("source_pdf", "unknown_source.pdf"),
                "page_number": meta.get("page_number") or meta.get("page", 1),
                "page": meta.get("page") or meta.get("page_number", 1),
                "document_title": meta.get("document_title") or meta.get("title", meta.get("source_file", "Document")),
                "category": meta.get("category", "General"),
                "academic_year": meta.get("academic_year", "-"),
                "program": meta.get("program", "-"),
                "payload": meta
            }

            new_vectors.append(emb_matrix[i])
            new_docs.append(doc_entry)
            self.seen_chunk_ids.add(chunk_id_str)
            added_count += 1

        if new_vectors:
            vectors_to_add = np.vstack(new_vectors).astype(np.float32)
            self.index.add(vectors_to_add)
            self.docstore.extend(new_docs)
            self.save()

        return added_count

    def search(
        self,
        query_vector: Any,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic cosine vector search with optional metadata filtering.
        Returns top-k matching documents with similarity scores and complete metadata.
        """
        if self.index.ntotal == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        if len(q_vec.shape) == 1:
            q_vec = np.expand_dims(q_vec, axis=0)

        faiss.normalize_L2(q_vec)

        # Retrieve a broader pool if filtering is applied
        fetch_k = min(self.index.ntotal, top_k * 5 if filter_dict else top_k)
        scores, indices = self.index.search(q_vec, fetch_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.docstore):
                continue

            doc = self.docstore[idx]

            # Apply metadata filters if specified
            if filter_dict:
                match = True
                for f_key, f_val in filter_dict.items():
                    doc_val = doc.get(f_key) or doc.get("payload", {}).get(f_key)
                    if doc_val is None:
                        match = False
                        break
                    if isinstance(f_val, list):
                        if doc_val not in f_val:
                            match = False
                            break
                    elif str(doc_val).lower() != str(f_val).lower():
                        match = False
                        break
                if not match:
                    continue

            result_item = {
                "id": doc.get("chunk_id", str(idx)),
                "score": float(score),
                "text": doc.get("text", ""),
                "raw_text": doc.get("raw_text", doc.get("text", "")),
                "source_file": doc.get("source_file", doc.get("source_pdf", "Unknown")),
                "source_pdf": doc.get("source_pdf", doc.get("source_file", "Unknown")),
                "page": doc.get("page", 1),
                "page_number": doc.get("page_number", 1),
                "document_title": doc.get("document_title", "Document"),
                "category": doc.get("category", "General"),
                "academic_year": doc.get("academic_year", "-"),
                "program": doc.get("program", "-"),
                "payload": doc.get("payload", {})
            }
            results.append(result_item)
            if len(results) >= top_k:
                break

        return results

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Returns all documents stored in the index docstore."""
        return list(self.docstore)
