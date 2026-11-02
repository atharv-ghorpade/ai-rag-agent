import argparse
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from enterprise_rag.src.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    COLLECTION_NAME,
    VECTOR_STORE_TYPE,
    QA_DATASET_PATH
)

DEFAULT_VIT_DATA_DIR = r"D:\Downloads\VIT_Data__for_AI_Project_"


def print_banner():
    banner = f"""
======================================================================
  ENTERPRISE RAG INTELLIGENCE SYSTEM
  Store: {VECTOR_STORE_TYPE.upper()} | Embedding: {DEFAULT_EMBEDDING_MODEL}
  Person 1 (Ingestion) | Person 2 (FAISS/Qdrant & Rerank) | Person 3 (Grounded Gen)
======================================================================
"""
    print(banner)


def run_demo(args):
    """Runs a complete end-to-end demonstration using sample enterprise policies."""
    from enterprise_rag.src.pipeline import EnterpriseRAGPipeline
    print_banner()
    print("[1/3] Initializing pipeline...")
    pipeline = EnterpriseRAGPipeline()

    sample_file = PROJECT_ROOT / "data" / "sample_documents.json"
    print(f"\n[2/3] Indexing documents from {sample_file}...")
    pipeline.index_from_json(str(sample_file))

    queries = [
        "How many vacation days do employees get?",
        "What are the rules regarding remote work and working from home?",
        "How do I submit business travel expenses?",
        "What is the company policy on password expiration and MFA?"
    ]

    print("\n[3/3] Running sample queries & comparing Vector Search vs Reranker:\n")
    for q in queries:
        print("=" * 70)
        print(f"QUERY: \"{q}\"")
        print("=" * 70)
        res = pipeline.query_with_comparison(q, retrieval_k=5, final_k=3)

        print("\n--- Initial Vector Search (Cosine Similarity) ---")
        for i, item in enumerate(res["vector_search_top_k"][:3], 1):
            print(f"  {i}. [Score: {item['score']:.4f}] {item['text']}")

        print("\n--- After Cross-Encoder Reranking ---")
        for item in res["reranked_top_k"]:
            print(f"  {item['rank']}. [Score: {item['score']:.4f}] {item['text']}")
        print()


def run_ingest(args):
    """Ingests documents from directory or cached JSONs into the active vector database."""
    from enterprise_rag.src.pipeline import EnterpriseRAGPipeline
    print_banner()
    pipeline = EnterpriseRAGPipeline()

    output_json = str(PROJECT_ROOT / "data" / "vit_chunks.json")

    if args.use_cache:
        cache_dir = str(PROJECT_ROOT / "data" / "parsed_cache")
        print(f"Starting Ingestion from Parsed Cache: {cache_dir}")
        count = pipeline.ingest_from_parsed_cache(cache_dir=cache_dir, output_json=output_json)
    else:
        directory = args.dir or DEFAULT_VIT_DATA_DIR
        print(f"Starting Ingestion from Directory: {directory}")
        count = pipeline.ingest_directory(directory_path=directory, output_json=output_json)

    print(f"\n[Success] Ingestion complete! {count} chunks indexed into {pipeline.vector_store_type.upper()}.")


def run_ask(args):
    """End-to-End RAG: Retrieves relevant chunks, reranks, and synthesizes grounded answer."""
    from enterprise_rag.src.pipeline import EnterpriseRAGPipeline
    print_banner()
    pipeline = EnterpriseRAGPipeline()

    filter_dict = {}
    if args.category:
        filter_dict["category"] = args.category
    if args.document:
        filter_dict["source_pdf"] = args.document

    res = pipeline.ask(
        question=args.question,
        retrieval_k=args.retrieval_k,
        final_k=args.final_k,
        filter_dict=filter_dict if filter_dict else None,
        use_qa_boost=not args.no_qa_boost
    )

    print("\n" + "=" * 70)
    print(f"QUESTION: \"{args.question}\"")
    print("=" * 70)
    if res.get("matched_qa"):
        top_match = res["matched_qa"][0]
        print(f"[QA-Boost] Matched Reference Question: \"{top_match['question']}\" (sim: {top_match.get('similarity', 0):.3f})")
    print("\n" + res["answer"])
    print("\n" + "-" * 70)
    print("CITATIONS & VERIFIED SOURCES:")
    for i, c in enumerate(res["citations"], 1):
        print(f"  [{i}] File: {c['source_file']} | Page: {c['page']} | Confidence: {c['score']:.4f}")
    print("=" * 70)


def run_hybrid(args):
    """Executes hybrid retrieval combining semantic vector search, question similarity, and metadata filtering."""
    from enterprise_rag.src.pipeline import EnterpriseRAGPipeline
    print_banner()
    pipeline = EnterpriseRAGPipeline()

    filter_dict = {}
    if args.category:
        filter_dict["category"] = args.category
    if args.document:
        filter_dict["source_pdf"] = args.document

    res = pipeline.hybrid_search(
        query=args.query,
        retrieval_k=args.retrieval_k,
        final_k=args.final_k,
        filter_dict=filter_dict if filter_dict else None,
        use_qa_boost=not args.no_qa_boost
    )

    print("\n" + "=" * 70)
    print(f"HYBRID RETRIEVAL FOR: \"{args.query}\"")
    print("=" * 70)
    if res.get("matched_qa"):
        print("\n[1] Top Matched Reference Questions:")
        for q in res["matched_qa"]:
            print(f"  * \"{q['question']}\" | Sim: {q.get('similarity', 0):.4f} | Target: {q.get('document')}")

    print(f"\n[2] Top Reranked Document Chunks (Selected from {res['total_candidates']} candidates):")
    for c in res["chunks"]:
        print(f"\n  [Rank {c['rank']}] Score: {c['rerank_score']:.4f} | File: {c['source_file']} (P.{c['page']})")
        print(f"  Title: {c.get('document_title', 'Document')} | Category: {c.get('category', 'General')}")
        print(f"  Snippet: {c['raw_text'][:160]}...")
    print("=" * 70)


def run_evaluate_qa(args):
    """Evaluates the pipeline on the QA dataset (Hit@K, Precision@K, Recall@K, Similarity, Latency)."""
    from enterprise_rag.src.evaluator import RAGEvaluator
    from enterprise_rag.src.pipeline import EnterpriseRAGPipeline
    print_banner()
    pipeline = EnterpriseRAGPipeline()
    evaluator = RAGEvaluator(pipeline=pipeline)

    summary = evaluator.evaluate_dataset(
        qa_dataset_path=args.dataset or QA_DATASET_PATH,
        max_samples=args.samples,
        retrieval_k=args.retrieval_k,
        final_k=args.final_k
    )
    evaluator.print_evaluation_report(summary)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Detailed evaluation metrics saved to: {out_p}")


def run_interactive(args):
    """Starts an interactive Q&A assistant session."""
    from enterprise_rag.src.pipeline import EnterpriseRAGPipeline
    print_banner()
    print("Starting interactive Enterprise RAG session...")
    pipeline = EnterpriseRAGPipeline()

    total = pipeline.get_total_chunks()
    if total == 0:
        print("\nVector database is currently empty.")
        cache_dir = str(PROJECT_ROOT / "data" / "parsed_cache")
        print(f"Auto-ingesting documents from: {cache_dir}...")
        pipeline.ingest_from_parsed_cache(cache_dir=cache_dir)

    print("\nReady! Type your question below (or 'exit' / 'quit' to stop):\n")
    while True:
        try:
            q = input("\nEnter Question > ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("Exiting interactive session.")
                break

            ans_res = pipeline.ask(q, retrieval_k=10, final_k=3)
            print("\n" + ans_res["answer"])
            print("\nCitations:")
            for c in ans_res["citations"]:
                print(f"  * {c['source_file']} (Page {c['page']})")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def run_benchmark(args):
    """Person 2: Evaluates retrieval performance."""
    if args.qa:
        run_evaluate_qa(args)
        return

    from enterprise_rag.src.evaluator import RAGEvaluator
    from enterprise_rag.src.pipeline import EnterpriseRAGPipeline
    print_banner()
    pipeline = EnterpriseRAGPipeline()
    evaluator = RAGEvaluator(pipeline=pipeline)
    summary = evaluator.run_benchmark()
    evaluator.print_evaluation_report(summary)


def run_serve(args):
    """Starts the FastAPI Web Application & REST Server."""
    import uvicorn
    print_banner()
    print(f"Starting Enterprise RAG Web Dashboard & API on http://{args.host}:{args.port}")
    if args.reload:
        uvicorn.run(
            "enterprise_rag.src.api:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(PROJECT_ROOT / "enterprise_rag")]
        )
    else:
        uvicorn.run("enterprise_rag.src.api:app", host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise RAG System (Ingestion, Hybrid Search, QA-Aware Retrieval, Evaluation, Web UI)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    ingest_parser.add_argument("--dir", "-d", type=str, default=DEFAULT_VIT_DATA_DIR, help="Path to PDF directory")
    ingest_parser.add_argument("--use-cache", "-c", action="store_true", default=True, help="Ingest from data/parsed_cache/")
    ingest_parser.set_defaults(func=run_ingest)

    # Ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question with grounded answer & citations")
    ask_parser.add_argument("question", type=str, help="Question to answer")
    ask_parser.add_argument("--retrieval-k", "-r", type=int, default=10, help="Candidate chunks to retrieve")
    ask_parser.add_argument("--final-k", "-k", type=int, default=4, help="Top reranked chunks for generation")
    ask_parser.add_argument("--category", type=str, default=None, help="Filter by category")
    ask_parser.add_argument("--document", type=str, default=None, help="Filter by document name")
    ask_parser.add_argument("--no-qa-boost", action="store_true", help="Disable QA similarity boost")
    ask_parser.set_defaults(func=run_ask)

    # Hybrid Search
    hybrid_parser = subparsers.add_parser("hybrid", help="Run hybrid retrieval (semantic + QA similarity + metadata filter)")
    hybrid_parser.add_argument("query", type=str, help="Search query")
    hybrid_parser.add_argument("--retrieval-k", "-r", type=int, default=10, help="Retrieval candidates")
    hybrid_parser.add_argument("--final-k", "-k", type=int, default=4, help="Top reranked candidates")
    hybrid_parser.add_argument("--category", type=str, default=None, help="Filter by category")
    hybrid_parser.add_argument("--document", type=str, default=None, help="Filter by document name")
    hybrid_parser.add_argument("--no-qa-boost", action="store_true", help="Disable QA similarity boost")
    hybrid_parser.set_defaults(func=run_hybrid)

    # Evaluate QA Dataset
    eval_parser = subparsers.add_parser("evaluate-qa", help="Evaluate pipeline on the 115-item QA dataset")
    eval_parser.add_argument("--dataset", type=str, default=None, help="Path to QA dataset JSON")
    eval_parser.add_argument("--samples", "-s", type=int, default=None, help="Limit number of queries to evaluate")
    eval_parser.add_argument("--retrieval-k", "-r", type=int, default=10, help="Retrieval candidates")
    eval_parser.add_argument("--final-k", "-k", type=int, default=4, help="Top reranked candidates")
    eval_parser.add_argument("--output", "-o", type=str, default="data/evaluation_results.json", help="Output JSON path")
    eval_parser.set_defaults(func=run_evaluate_qa)

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run quantitative evaluation")
    bench_parser.add_argument("--qa", action="store_true", help="Run on full QA dataset")
    bench_parser.add_argument("--samples", "-s", type=int, default=None, help="Limit sample count")
    bench_parser.add_argument("--retrieval-k", "-r", type=int, default=10, help="Retrieval candidates")
    bench_parser.add_argument("--final-k", "-k", type=int, default=4, help="Reranked candidates")
    bench_parser.add_argument("--output", "-o", type=str, default="data/evaluation_results.json", help="Output JSON path")
    bench_parser.set_defaults(func=run_benchmark)

    # Serve (FastAPI + Web Dashboard)
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST server & Web Dashboard")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument("--reload", action="store_true", help="Enable live auto-reloading")
    serve_parser.set_defaults(func=run_serve)

    # Interactive
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive Q&A assistant prompt")
    interactive_parser.set_defaults(func=run_interactive)

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Run sample enterprise policy demo")
    demo_parser.set_defaults(func=run_demo)

    args = parser.parse_args()

    if not args.command:
        run_ask(argparse.Namespace(
            question="Who is the Head of the Information Technology Department?",
            retrieval_k=10,
            final_k=4,
            category=None,
            document=None,
            no_qa_boost=False
        ))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
