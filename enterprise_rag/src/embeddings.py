import os

# Store Hugging Face models on D: drive
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/huggingface_cache/hub"

from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(self, model_name="BAAI/bge-m3"):
        print("Loading BGE-M3...")

        self.model = SentenceTransformer(model_name)

        print("BGE-M3 loaded successfully!")

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )

    def embed_query(self, query):
        return self.model.encode(
            query,
            normalize_embeddings=True
        )

    def get_dimension(self):
        return self.model.get_embedding_dimension()