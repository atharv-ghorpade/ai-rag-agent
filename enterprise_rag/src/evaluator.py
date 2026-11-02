import time
import json
import statistics
import re
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .config import ROOT_DIR, QA_DATASET_PATH
from .pipeline import EnterpriseRAGPipeline


DEFAULT_BENCHMARK_QUERIES = [
    {
        "query": "Who is the Head of the Information Technology Department?",
        "ground_truth_keywords": ["Vidya Chitre", "Head of the Information Technology Department"],
        "target_document": "Reflection_Jan2026.pdf",
        "category": "Fact"
    },
    {
        "query": "Who completed a PhD in Natural Language Processing?",
        "ground_truth_keywords": ["Rasika Ransing", "Natural Language Processing"],
        "target_document": "Reflection_Jan2026.pdf",
        "category": "Faculty Achievement"
    },
    {
        "query": "Who completed a PhD in Artificial Intelligence and Machine Learning?",
        "ground_truth_keywords": ["Neha Kudu", "Artificial Intelligence and Machine Learning"],
        "target_document": "Reflection_Jan2026.pdf",
        "category": "Faculty Achievement"
    },
    {
        "query": "What are the two categories under Institute Level admissions?",
        "ground_truth_keywords": ["Against CAP", "ACAP", "Institute Level", "20%"],
        "target_document": "Institute_Level_Admissions_Policy.pdf",
        "category": "Admissions"
    },
    {
        "query": "What is the total fee for MMS Institute Level admission?",
        "ground_truth_keywords": ["1,99,500", "199500", "tuition", "development"],
        "target_document": "Institute_Level_Admissions_Policy.pdf",
        "category": "Admissions Fees"
    },
    {
        "query": "What is the sanctioned intake for MMS for A.Y. 2022-23?",
        "ground_truth_keywords": ["139", "133", "mms"],
        "target_document": "Approval for MMS Admissions A.Y. 2022-23.pdf",
        "category": "Admissions Approval"
    },
    {
        "query": "What is the eligibility for Honours or Minor Degree in Computer Engineering?",
        "ground_truth_keywords": ["6.75", "CGPA", "80 credits"],
        "target_document": "BTech_Computer_Engineering_Honours_Minor_R2022.pdf",
        "category": "Curriculum"
    },
    {
        "query": "Who is the Head of Department of Computer Engineering?",
        "ground_truth_keywords": ["Ravindra Sangle", "Computer Engineering"],
        "target_document": "BYTES_Jan2026_Vol17_Issue02.pdf",
        "category": "Fact"
    }
]


class RAGEvaluator:
    """
    Quantitative Evaluation Framework for Enterprise RAG:
    - Top-k Retrieval Accuracy (Hit@1, Hit@3, Hit@5)
    - Precision@K and Recall@K for context relevance
    - Answer Correctness (token overlap & key fact verification)
    - Average Cosine Similarity Score
    - Query Latency (ms) for vector search vs cross-encoder reranking
    """

    def __init__(self, pipeline: Optional[EnterpriseRAGPipeline] = None):
        self.pipeline = pipeline or EnterpriseRAGPipeline()

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r"\b[A-Za-z0-9_]{3,}\b", text.lower())
        stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "what", "who", "which", "how"}
        return [w for w in words if w not in stopwords]

    def _calculate_answer_correctness(self, generated_answer: str, ground_truth: str) -> float:
        """Computes word-level F1 / overlap score between generated answer and ground truth."""
        gen_tokens = set(self._extract_keywords(generated_answer))
        gt_tokens = set(self._extract_keywords(ground_truth))

        if not gt_tokens:
            return 1.0 if not gen_tokens else 0.0

        overlap = gen_tokens.intersection(gt_tokens)
        if not overlap:
            return 0.0

        precision = len(overlap) / len(gen_tokens) if gen_tokens else 0.0
        recall = len(overlap) / len(gt_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_query(
        self,
        query_item: Dict[str, Any],
        retrieval_k: int = 10,
        final_k: int = 4
    ) -> Dict[str, Any]:
        query = query_item["question"] if "question" in query_item else query_item["query"]
        ground_truth = query_item.get("answer", "")
        target_doc = query_item.get("document") or query_item.get("target_document", "")
        keywords = query_item.get("ground_truth_keywords") or self._extract_keywords(ground_truth)

        t0 = time.perf_counter()
        hybrid_res = self.pipeline.hybrid_search(
            query=query,
            retrieval_k=retrieval_k,
            final_k=final_k
        )
        t_retrieval = (time.perf_counter() - t0) * 1000.0

        retrieved_chunks = hybrid_res["chunks"]

        # 1. Retrieval Accuracy (Hit@K) & Average Similarity
        target_rank = None
        similarities = []
        relevant_chunks_count = 0

        for rank, chunk in enumerate(retrieved_chunks, 1):
            chunk_doc = chunk.get("source_pdf") or chunk.get("source_file", "")
            chunk_text = (chunk.get("raw_text") or chunk.get("text", "")).lower()
            chunk_sim = chunk.get("cosine_similarity") if "cosine_similarity" in chunk else chunk.get("score", 0.0)
            similarities.append(chunk_sim)

            # Check document match or keyword overlap
            doc_matched = (target_doc.lower() in chunk_doc.lower()) if target_doc else False
            kw_matched = any(kw.lower() in chunk_text for kw in keywords) if keywords else False

            is_relevant = doc_matched or kw_matched
            if is_relevant:
                relevant_chunks_count += 1
                if target_rank is None:
                    target_rank = rank

        hit1 = 1 if target_rank == 1 else 0
        hit3 = 1 if (target_rank and target_rank <= 3) else 0
        hit5 = 1 if (target_rank and target_rank <= 5) else 0
        reciprocal_rank = (1.0 / target_rank) if target_rank else 0.0

        # 2. Context Relevance (Precision & Recall)
        precision_k = relevant_chunks_count / len(retrieved_chunks) if retrieved_chunks else 0.0
        recall_k = 1.0 if target_rank is not None else 0.0

        # 3. Answer Generation & Correctness
        t_gen_start = time.perf_counter()
        gen_res = self.pipeline.generator.generate_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            matched_qa=hybrid_res.get("matched_qa", [])
        )
        t_gen = (time.perf_counter() - t_gen_start) * 1000.0
        generated_answer = gen_res["answer"]

        answer_correctness = self._calculate_answer_correctness(generated_answer, ground_truth) if ground_truth else 1.0
        avg_similarity = statistics.mean(similarities) if similarities else 0.0

        return {
            "query": query,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "target_document": target_doc,
            "category": query_item.get("category", "General"),
            "difficulty": query_item.get("difficulty", "medium"),
            "target_rank": target_rank,
            "reciprocal_rank": reciprocal_rank,
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "precision@k": precision_k,
            "recall@k": recall_k,
            "answer_correctness": answer_correctness,
            "avg_similarity_score": avg_similarity,
            "retrieval_latency_ms": t_retrieval,
            "generation_latency_ms": t_gen,
            "total_latency_ms": t_retrieval + t_gen,
            "chunks_retrieved": len(retrieved_chunks)
        }

    def evaluate_dataset(
        self,
        qa_dataset_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        retrieval_k: int = 10,
        final_k: int = 4
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation over the provided QA dataset.
        """
        path = Path(qa_dataset_path or QA_DATASET_PATH)
        if not path.is_absolute():
            path = ROOT_DIR / qa_dataset_path

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        print(f"\nRunning evaluation on {len(data)} queries from {path.name}...", flush=True)
        results = []
        for i, item in enumerate(data, 1):
            res = self.evaluate_query(item, retrieval_k=retrieval_k, final_k=final_k)
            results.append(res)
            if i % 15 == 0 or i == len(data):
                print(f"  Processed [{i}/{len(data)}] queries...", flush=True)

        n = len(results)
        summary = {
            "total_queries_evaluated": n,
            "hit@1_accuracy": sum(r["hit@1"] for r in results) / n,
            "hit@3_accuracy": sum(r["hit@3"] for r in results) / n,
            "hit@5_accuracy": sum(r["hit@5"] for r in results) / n,
            "mean_reciprocal_rank_mrr": sum(r["reciprocal_rank"] for r in results) / n,
            "average_precision@k": sum(r["precision@k"] for r in results) / n,
            "average_recall@k": sum(r["recall@k"] for r in results) / n,
            "average_answer_correctness": sum(r["answer_correctness"] for r in results) / n,
            "average_similarity_score": statistics.mean(r["avg_similarity_score"] for r in results),
            "average_retrieval_latency_ms": statistics.mean(r["retrieval_latency_ms"] for r in results),
            "average_total_latency_ms": statistics.mean(r["total_latency_ms"] for r in results),
            "detailed_results": results
        }

        return summary

    def print_evaluation_report(self, summary: Dict[str, Any]):
        """Prints a comprehensive evaluation report table."""
        print("\n" + "=" * 80)
        print(" ENTERPRISE RAG SYSTEM EVALUATION REPORT (QA DATASET)")
        print("=" * 80)
        print(f"Total Queries Evaluated: {summary['total_queries_evaluated']}")
        print("-" * 80)
        print(f"{'Metric':<35} | {'Measured Score'}")
        print("-" * 80)
        print(f"{'Top-1 Retrieval Accuracy (Hit@1)':<35} | {summary['hit@1_accuracy']*100:>12.2f}%")
        print(f"{'Top-3 Retrieval Accuracy (Hit@3)':<35} | {summary['hit@3_accuracy']*100:>12.2f}%")
        print(f"{'Top-5 Retrieval Accuracy (Hit@5)':<35} | {summary['hit@5_accuracy']*100:>12.2f}%")
        print(f"{'Mean Reciprocal Rank (MRR)':<35} | {summary['mean_reciprocal_rank_mrr']:>13.4f}")
        print(f"{'Context Precision@K':<35} | {summary['average_precision@k']*100:>12.2f}%")
        print(f"{'Context Recall@K':<35} | {summary['average_recall@k']*100:>12.2f}%")
        print(f"{'Average Answer Correctness (F1)':<35} | {summary['average_answer_correctness']*100:>12.2f}%")
        print(f"{'Average Cosine Similarity':<35} | {summary['average_similarity_score']:>13.4f}")
        print(f"{'Average Retrieval Latency':<35} | {summary['average_retrieval_latency_ms']:>10.2f} ms")
        print(f"{'Average Total Latency':<35} | {summary['average_total_latency_ms']:>10.2f} ms")
        print("=" * 80 + "\n")

    def run_benchmark(self, queries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Executes comparative quantitative evaluation:
        Stage 1: Dense Vector Search (Cosine Retrieval)
        Stage 2: Cross-Encoder Neural Reranking
        Calculates Hit@1, Hit@3, Hit@5, MRR, and Latencies for both stages.
        """
        test_queries = queries or DEFAULT_BENCHMARK_QUERIES
        detailed_results = []

        dense_hit1_list = []
        dense_hit3_list = []
        dense_hit5_list = []
        dense_rr_list = []
        dense_lat_list = []

        rerank_hit1_list = []
        rerank_hit3_list = []
        rerank_hit5_list = []
        rerank_rr_list = []
        rerank_lat_list = []
        total_lat_list = []

        for q in test_queries:
            query_text = q.get("question") or q.get("query")
            target_doc = (q.get("document") or q.get("target_document") or "").lower()
            keywords = q.get("ground_truth_keywords") or self._extract_keywords(q.get("answer", ""))

            # --- Stage 1: Dense Vector Search ---
            t0 = time.perf_counter()
            dense_results = self.pipeline.retriever._dense_search(query_text, k=10)
            t_dense = (time.perf_counter() - t0) * 1000.0
            dense_lat_list.append(t_dense)

            # Determine Dense Rank
            dense_rank = None
            for r_idx, d_item in enumerate(dense_results, 1):
                src = (d_item.get("source_file") or d_item.get("source_pdf") or "").lower()
                txt = (d_item.get("raw_text") or d_item.get("text") or "").lower()
                doc_match = bool(target_doc and target_doc in src)
                kw_match = any(kw.lower() in txt for kw in keywords) if keywords else False
                if doc_match or kw_match:
                    dense_rank = r_idx
                    break

            d_hit1 = 1 if dense_rank == 1 else 0
            d_hit3 = 1 if (dense_rank and dense_rank <= 3) else 0
            d_hit5 = 1 if (dense_rank and dense_rank <= 5) else 0
            d_rr = (1.0 / dense_rank) if dense_rank else 0.0

            dense_hit1_list.append(d_hit1)
            dense_hit3_list.append(d_hit3)
            dense_hit5_list.append(d_hit5)
            dense_rr_list.append(d_rr)

            # --- Stage 2: Cross-Encoder Reranking ---
            candidate_texts = [item["text"] for item in dense_results]
            t1 = time.perf_counter()
            reranked_tuples = self.pipeline.reranker.rerank(query_text, candidate_texts, top_k=4)
            t_rerank = (time.perf_counter() - t1) * 1000.0
            t_total = t_dense + t_rerank

            rerank_lat_list.append(t_rerank)
            total_lat_list.append(t_total)

            text_to_item = {item["text"]: item for item in dense_results}
            rerank_rank = None
            for r_idx, (doc_text, score) in enumerate(reranked_tuples, 1):
                orig_item = text_to_item.get(doc_text, {})
                src = (orig_item.get("source_file") or orig_item.get("source_pdf") or "").lower()
                txt = (orig_item.get("raw_text") or doc_text).lower()
                doc_match = bool(target_doc and target_doc in src)
                kw_match = any(kw.lower() in txt for kw in keywords) if keywords else False
                if doc_match or kw_match:
                    rerank_rank = r_idx
                    break

            r_hit1 = 1 if rerank_rank == 1 else 0
            r_hit3 = 1 if (rerank_rank and rerank_rank <= 3) else 0
            r_hit5 = 1 if (rerank_rank and rerank_rank <= 5) else 0
            r_rr = (1.0 / rerank_rank) if rerank_rank else 0.0

            rerank_hit1_list.append(r_hit1)
            rerank_hit3_list.append(r_hit3)
            rerank_hit5_list.append(r_hit5)
            rerank_rr_list.append(r_rr)

            detailed_results.append({
                "query": query_text,
                "dense_rank": dense_rank,
                "dense_latency_ms": t_dense,
                "rerank_rank": rerank_rank,
                "rerank_latency_ms": t_rerank,
                "total_latency_ms": t_total,
                "target_document": target_doc
            })

        n = len(test_queries) or 1
        summary = {
            "total_queries_evaluated": n,
            "dense_search": {
                "hit@1": sum(dense_hit1_list) / n,
                "hit@3": sum(dense_hit3_list) / n,
                "hit@5": sum(dense_hit5_list) / n,
                "mrr": sum(dense_rr_list) / n,
                "avg_latency_ms": statistics.mean(dense_lat_list) if dense_lat_list else 0.0
            },
            "cross_encoder_reranked": {
                "hit@1": sum(rerank_hit1_list) / n,
                "hit@3": sum(rerank_hit3_list) / n,
                "hit@5": sum(rerank_hit5_list) / n,
                "mrr": sum(rerank_rr_list) / n,
                "avg_rerank_latency_ms": statistics.mean(rerank_lat_list) if rerank_lat_list else 0.0,
                "avg_total_latency_ms": statistics.mean(total_lat_list) if total_lat_list else 0.0
            },
            "detailed_results": detailed_results,
            # Backwards-compatible root fields
            "hit@1_accuracy": sum(rerank_hit1_list) / n,
            "hit@3_accuracy": sum(rerank_hit3_list) / n,
            "hit@5_accuracy": sum(rerank_hit5_list) / n,
            "mean_reciprocal_rank_mrr": sum(rerank_rr_list) / n,
            "average_total_latency_ms": statistics.mean(total_lat_list) if total_lat_list else 0.0
        }
        return summary
