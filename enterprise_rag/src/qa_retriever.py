import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from .config import ROOT_DIR, QA_DATASET_PATH, QA_SIMILARITY_THRESHOLD, VECTOR_SIZE
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class QAIndexManager:
    """
    Manages a specialized FAISS semantic index for verified Question-Answer pairs.
    
    Enterprise Utility:
    In production enterprise RAG systems, organizations maintain verified FAQs,
    historical audit questions, or curated Q&A pairs. This component embeds and
    indexes all reference questions. During user inference, query-to-question similarity
    is evaluated to boost recall, provide validated answer hints, and perform hybrid retrieval.
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        qa_dataset_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.embedder = embedding_manager or EmbeddingManager()
        self.qa_dataset_path = Path(qa_dataset_path or QA_DATASET_PATH)
        self.cache_dir = Path(cache_dir or (ROOT_DIR / "data" / "qa_index"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.cache_dir / "qa_index.faiss"
        self.meta_file = self.cache_dir / "qa_metadata.json"

        self.qa_pairs: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None

        self._load_or_build_index()

    def _load_or_build_index(self):
        """Loads existing QA FAISS index or builds it from qa_dataset.json."""
        if self.index_file.exists() and self.meta_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.qa_pairs = json.load(f)
                logger.info(f"Loaded QA index with {len(self.qa_pairs)} question pairs.")
                return
            except Exception as e:
                logger.warning(f"Could not load cached QA index ({e}). Rebuilding...")

        self.rebuild_index()

    def rebuild_index(self) -> int:
        """Embeds all questions in the QA dataset and creates the FAISS index."""
        if not self.qa_dataset_path.exists():
            logger.warning(f"QA dataset not found at {self.qa_dataset_path}")
            return 0

        with open(self.qa_dataset_path, "r", encoding="utf-8") as f:
            self.qa_pairs = json.load(f)

        if not self.qa_pairs:
            return 0

        questions = [item["question"] for item in self.qa_pairs]
        embeddings = self.embedder.embed_documents(questions)

        emb_matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb_matrix)

        dim = emb_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb_matrix)

        # Persist index and metadata
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Built and saved QA Index with {len(self.qa_pairs)} items.")
        return len(self.qa_pairs)

    def search_similar_questions(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = QA_SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Finds pre-existing verified questions that are semantically similar to the incoming user query.
        Returns matches exceeding the similarity threshold.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = self.embedder.embed_query(query)
        q_vec = np.array(q_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.qa_pairs):
                continue
            sim = float(score)
            if sim >= threshold:
                item = dict(self.qa_pairs[idx])
                item["similarity"] = sim
                matches.append(item)

        return matches
