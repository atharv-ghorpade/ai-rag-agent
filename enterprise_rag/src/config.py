import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

# Hugging Face Cache Setup
HF_HOME = os.getenv("HF_HOME", "D:/huggingface_cache")
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ensure cache directory exists if possible
try:
    os.makedirs(HF_HOME, exist_ok=True)
except Exception:
    # If D: drive is unavailable or read-only, fallback to local cache
    fallback = str(ROOT_DIR / "cache")
    os.environ["HF_HOME"] = fallback
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(fallback, "hub")
    os.makedirs(fallback, exist_ok=True)

# Model Settings
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_documents")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))

# Vector Store Engine ('faiss' or 'qdrant')
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()
FAISS_STORAGE_PATH = os.getenv("FAISS_STORAGE_PATH", str(ROOT_DIR / "data" / "faiss_index"))

# Qdrant Settings (if using Qdrant)
QDRANT_MODE = os.getenv("QDRANT_MODE", "local").lower()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", str(ROOT_DIR / "data" / "qdrant_storage"))

# Chunking & QA Dataset Settings
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
QA_DATASET_PATH = os.getenv("QA_DATASET_PATH", str(ROOT_DIR / "data" / "qa_dataset.json"))
QA_SIMILARITY_THRESHOLD = float(os.getenv("QA_SIMILARITY_THRESHOLD", "0.68"))
