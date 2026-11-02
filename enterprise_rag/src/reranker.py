import os

cache_dir = os.environ.get("HF_HOME", "D:/huggingface_cache")
try:
    os.makedirs(cache_dir, exist_ok=True)
except Exception:
    pass

from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name=None
    ):
        model_name = model_name or os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        print(f"Loading reranker model: {model_name}...")

        self.model = CrossEncoder(
            model_name,
            cache_folder=cache_dir
        )

        print(f"Reranker model '{model_name}' loaded successfully!")

    def rerank(self, query, documents, top_k=3):
        if not documents:
            return []

        pairs = [
            [query, document]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]