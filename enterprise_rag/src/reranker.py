import os

from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3"
    ):
        print("Loading reranker...")

        self.model = CrossEncoder(
            model_name,
            cache_folder="D:/huggingface_cache"
        )

        print("Reranker loaded successfully!")

    def rerank(self, query, documents, top_k=3):

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