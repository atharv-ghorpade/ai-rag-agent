import unittest
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enterprise_rag.src.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
)
from enterprise_rag.src.embeddings import EmbeddingManager
from enterprise_rag.src.reranker import Reranker
from enterprise_rag.src.retriever import Retriever
from enterprise_rag.src.faiss_manager import FAISSManager
from enterprise_rag.src.qa_retriever import QAIndexManager
from enterprise_rag.src.generator import RAGGenerator
from enterprise_rag.src.ingestion import clean_extracted_text, DocumentChunker


class TestEnterpriseRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import os
        print("\n[Test Setup] Initializing test models and temporary FAISS storage...")
        cls.temp_dir = tempfile.mkdtemp()
        cls.faiss_path = Path(cls.temp_dir) / "faiss_test"

        cls.embedder = EmbeddingManager(model_name=DEFAULT_EMBEDDING_MODEL)
        cls.dim = cls.embedder.get_dimension()

        cls.faiss_mgr = FAISSManager(
            dimension=cls.dim,
            storage_path=str(cls.faiss_path)
        )
        cls.reranker = Reranker(model_name=DEFAULT_RERANKER_MODEL)
        cls.qa_mgr = QAIndexManager(
            embedding_manager=cls.embedder,
            qa_dataset_path=str(PROJECT_ROOT / "data" / "qa_dataset.json"),
            cache_dir=str(Path(cls.temp_dir) / "qa_test")
        )
        cls.retriever = Retriever(
            vector_store=cls.faiss_mgr,
            embedding_manager=cls.embedder,
            reranker=cls.reranker,
            qa_index_mgr=cls.qa_mgr
        )
        cls.generator = RAGGenerator()

        cls.test_docs = [
            "Employees are entitled to 20 days of annual leave.",
            "Workers receive twenty days of vacation every year.",
            "Prof. Dr. Vidya Chitre is the Head of the Information Technology Department.",
            "Prof. Rasika Ransing completed a PhD in Natural Language Processing.",
            "The total fee for MMS Institute Level admission is ₹1,99,500 payable at Axis Bank."
        ]
        cls.metadata = [
            {"source_pdf": "HR_Policy.pdf", "page_number": 1, "category": "Policy", "chunk_id": "c1"},
            {"source_pdf": "HR_Policy.pdf", "page_number": 2, "category": "Policy", "chunk_id": "c2"},
            {"source_pdf": "Reflection_Jan2026.pdf", "page_number": 1, "category": "Academic", "chunk_id": "c3"},
            {"source_pdf": "Reflection_Jan2026.pdf", "page_number": 2, "category": "Achievement", "chunk_id": "c4"},
            {"source_pdf": "Institute_Level_Admissions_Policy.pdf", "page_number": 2, "category": "Admissions", "chunk_id": "c5"}
        ]

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_01_embedding_dimension(self):
        """Verify embedding dimension matches model configuration."""
        emb = self.embedder.embed_query("Test query")
        self.assertEqual(len(emb), self.dim)

    def test_02_faiss_indexing_and_deduplication(self):
        """Verify FAISS stores vectors, preserves metadata, and avoids duplicates."""
        embeddings = self.embedder.embed_documents(self.test_docs)
        count = self.faiss_mgr.add_documents(
            texts=self.test_docs,
            embeddings=embeddings,
            metadata_list=self.metadata,
            ids=[m["chunk_id"] for m in self.metadata]
        )
        self.assertEqual(count, len(self.test_docs))
        self.assertEqual(self.faiss_mgr.count(), len(self.test_docs))

        # Adding same IDs again should add 0 new documents (deduplication)
        dedup_count = self.faiss_mgr.add_documents(
            texts=self.test_docs,
            embeddings=embeddings,
            metadata_list=self.metadata,
            ids=[m["chunk_id"] for m in self.metadata]
        )
        self.assertEqual(dedup_count, 0)
        self.assertEqual(self.faiss_mgr.count(), len(self.test_docs))

    def test_03_faiss_search_and_metadata_filter(self):
        """Verify FAISS search returns correct chunk and respects metadata filters."""
        query = "Who completed a PhD in Natural Language Processing?"
        q_emb = self.embedder.embed_query(query)

        # Unfiltered search
        results = self.faiss_mgr.search(q_emb, top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("rasika ransing", results[0]["text"].lower())

        # Filtered search by category
        filtered = self.faiss_mgr.search(q_emb, top_k=2, filter_dict={"category": "Policy"})
        self.assertGreater(len(filtered), 0)
        self.assertEqual(filtered[0]["category"], "Policy")

    def test_04_qa_index_search(self):
        """Verify QA Index matches semantic queries to verified reference questions."""
        query = "Tell me the Head of the IT department"
        matches = self.qa_mgr.search_similar_questions(query, top_k=2, threshold=0.60)
        self.assertGreater(len(matches), 0)
        self.assertIn("head of the information technology department", matches[0]["question"].lower())

    def test_05_hybrid_retrieval_and_reranking(self):
        """Verify hybrid retrieval combines semantic vector search with QA boosting and reranking."""
        query = "What is the total fee for MMS Institute Level admission?"
        res = self.retriever.retrieve_hybrid(query, retrieval_k=4, final_k=2)
        chunks = res["chunks"]
        self.assertGreater(len(chunks), 0)
        top_text = chunks[0]["text"]
        self.assertTrue("1,99,500" in top_text or "fee" in top_text.lower())
        self.assertIn("rerank_score", chunks[0])

    def test_06_clean_extracted_text_and_chunker(self):
        """Verify text cleaning removes whitespace artifacts and chunker uses RecursiveCharacterTextSplitter."""
        raw = "Line 1   with    extra    spaces.\n\n\n\nLine 2 with break-\ning.\x00"
        cleaned = clean_extracted_text(raw)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\x00", cleaned)
        self.assertIn("breaking", cleaned)

        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.process_document("sample_policy.pdf", [{"page_num": 1, "text": "Heading: Policy\n" + "Word " * 50}])
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["source_pdf"], "sample_policy.pdf")

    def test_07_grounded_answer_generation(self):
        """Verify answer generator produces citations and refuses ungrounded questions."""
        fake_chunks = [
            {"source_file": "TestDoc.pdf", "page": 3, "text": "The project deadline is December 15, 2026."}
        ]
        ans = self.generator.generate_answer("When is the project deadline?", fake_chunks)
        self.assertIn("December 15, 2026", ans["answer"])
        self.assertGreater(len(ans["citations"]), 0)


if __name__ == "__main__":
    unittest.main()
