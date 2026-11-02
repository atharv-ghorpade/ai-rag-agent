import os

# Store Hugging Face models on D: drive (or configured HF_HOME)
cache_dir = os.environ.get("HF_HOME", "D:/huggingface_cache")
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
try:
    os.makedirs(cache_dir, exist_ok=True)
except Exception:
    pass

from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(self, model_name=None):
        model_name = model_name or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        print(f"Loading embedding model: {model_name}...")

        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

        print(f"Embedding model '{model_name}' loaded successfully!")

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