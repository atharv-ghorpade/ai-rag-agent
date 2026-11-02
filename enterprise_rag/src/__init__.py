def __getattr__(name: str):
    if name == "EmbeddingManager":
        from .embeddings import EmbeddingManager
        return EmbeddingManager
    elif name == "Reranker":
        from .reranker import Reranker
        return Reranker
    elif name == "Retriever":
        from .retriever import Retriever
        return Retriever
    elif name == "EnterpriseRAGPipeline":
        from .pipeline import EnterpriseRAGPipeline
        return EnterpriseRAGPipeline
    elif name == "FAISSManager":
        from .faiss_manager import FAISSManager
        return FAISSManager
    elif name == "QdrantManager":
        from .qdrant_manager import QdrantManager
        return QdrantManager
    elif name == "QAIndexManager":
        from .qa_retriever import QAIndexManager
        return QAIndexManager
    elif name == "RAGGenerator":
        from .generator import RAGGenerator
        return RAGGenerator
    elif name == "RAGEvaluator":
        from .evaluator import RAGEvaluator
        return RAGEvaluator
    elif name == "IngestionPipeline":
        from .ingestion import IngestionPipeline
        return IngestionPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "EmbeddingManager",
    "Reranker",
    "Retriever",
    "EnterpriseRAGPipeline",
    "FAISSManager",
    "QdrantManager",
    "QAIndexManager",
    "RAGGenerator",
    "RAGEvaluator",
    "IngestionPipeline"
]
