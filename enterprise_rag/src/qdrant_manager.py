import logging
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .config import QDRANT_MODE, QDRANT_URL, QDRANT_STORAGE_PATH, COLLECTION_NAME, VECTOR_SIZE

logger = logging.getLogger(__name__)


def create_qdrant_client(
    mode: Optional[str] = None,
    url: Optional[str] = None,
    storage_path: Optional[str] = None
) -> QdrantClient:
    """
    Factory function to initialize a QdrantClient based on mode:
    - 'server': Connects to a Qdrant server at `url` (e.g. http://localhost:6333)
    - 'local': Uses local embedded file-based storage at `storage_path`
    - 'memory': In-memory ephemeral storage for testing
    """
    mode = (mode or QDRANT_MODE).lower()
    url = url or QDRANT_URL
    storage_path = storage_path or QDRANT_STORAGE_PATH

    if mode == "server":
        try:
            print(f"Connecting to Qdrant server at {url}...")
            client = QdrantClient(url=url, timeout=3.0)
            client.get_collections()
            print("Successfully connected to Qdrant server!")
            return client
        except Exception as e:
            print(f"Notice: Could not connect to Qdrant server at {url} ({e}).")
            print(f"Falling back to local embedded storage at: {storage_path}")
            return QdrantClient(path=storage_path)

    elif mode == "memory":
        print("Using in-memory Qdrant storage (:memory:)...")
        return QdrantClient(":memory:")

    else:
        print(f"Using local embedded Qdrant storage at: {storage_path}")
        return QdrantClient(path=storage_path)


class QdrantManager:
    """
    Manager for interacting with Qdrant vector database:
    collection management, indexing, document upserting, and retrieval.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE
    ):
        self.client = client or create_qdrant_client()
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.ensure_collection()

    def ensure_collection(self, recreate: bool = False):
        """Ensures the collection exists with the specified dimension."""
        collections = [col.name for col in self.client.get_collections().collections]

        if recreate and self.collection_name in collections:
            self.client.delete_collection(collection_name=self.collection_name)
            collections.remove(self.collection_name)

        if self.collection_name not in collections:
            print(f"Creating collection '{self.collection_name}' (dim={self.vector_size})...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created successfully!")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def upsert_documents(
        self,
        texts: List[str],
        embeddings: Any,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[Any]] = None
    ) -> int:
        """
        Upsert a list of document texts with their corresponding embeddings and optional metadata.
        """
        points = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            raw_id = ids[i] if ids and i < len(ids) else i + 1
            payload = {"text": text, "chunk_id": str(raw_id)}
            if metadata_list and i < len(metadata_list):
                payload.update(metadata_list[i])

            # Ensure point ID is valid for Qdrant (either int or valid UUID string)
            if isinstance(raw_id, int):
                point_id = raw_id
            else:
                try:
                    uuid.UUID(str(raw_id))
                    point_id = str(raw_id)
                except Exception:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))

            vec_data = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec_data,
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Upserted {len(points)} points into '{self.collection_name}'.")
        return len(points)

    def count(self) -> int:
        """Returns the number of points stored in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
