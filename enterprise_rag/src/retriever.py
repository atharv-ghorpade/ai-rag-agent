class Retriever:

    def __init__(
        self,
        qdrant_client,
        embedding_manager,
        reranker,
        collection_name="enterprise_documents"
    ):
        self.qdrant = qdrant_client
        self.embedder = embedding_manager
        self.reranker = reranker
        self.collection_name = collection_name

    def retrieve(
        self,
        query,
        retrieval_k=5,
        final_k=3
    ):
        # 1. Convert query into embedding
        query_embedding = self.embedder.embed_query(query)

        # 2. Search Qdrant
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=retrieval_k
        )

        # 3. Extract documents
        documents = [
            result.payload["text"]
            for result in results.points
        ]

        # 4. Rerank
        reranked = self.reranker.rerank(
            query,
            documents,
            top_k=final_k
        )

        return reranked