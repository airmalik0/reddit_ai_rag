"""Vector store operations using Qdrant with Hybrid Search."""

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

from qdrant_client import QdrantClient

from .config import (
    EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


# ---------------------------------------------------------------------------
# Singletons for expensive resources
# ---------------------------------------------------------------------------

_qdrant_client: QdrantClient | None = None
_dense_embeddings: OpenAIEmbeddings | None = None
_sparse_embeddings: FastEmbedSparse | None = None
_vector_store: dict[str, QdrantVectorStore] = {}


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client (singleton).

    Returns:
        QdrantClient instance
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


def get_dense_embeddings() -> OpenAIEmbeddings:
    """Get dense embeddings model (singleton).

    Returns:
        OpenAIEmbeddings instance
    """
    global _dense_embeddings
    if _dense_embeddings is None:
        _dense_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _dense_embeddings


def get_sparse_embeddings() -> FastEmbedSparse:
    """Get sparse embeddings model (singleton).

    Returns:
        FastEmbedSparse instance
    """
    global _sparse_embeddings
    if _sparse_embeddings is None:
        _sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)
    return _sparse_embeddings


def get_vector_store(collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:
    """Get vector store connected to existing Qdrant collection (singleton per collection).

    Args:
        collection_name: Name of the Qdrant collection

    Returns:
        QdrantVectorStore instance with hybrid search enabled
    """
    global _vector_store
    if collection_name not in _vector_store:
        _vector_store[collection_name] = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=collection_name,
            embedding=get_dense_embeddings(),
            sparse_embedding=get_sparse_embeddings(),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )
    return _vector_store[collection_name]


def search_similar(query: str, k: int = 10) -> list[dict]:
    """Search for similar documents.

    Args:
        query: Search query
        k: Number of results

    Returns:
        List of dicts with post_id and score
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(query, k=k)

    seen_post_ids = set()
    unique_results = []

    for doc, score in results:
        post_id = doc.metadata.get('post_id')
        if post_id and post_id not in seen_post_ids:
            seen_post_ids.add(post_id)
            unique_results.append({
                'post_id': post_id,
                'score': score,
                'chunk_content': doc.page_content
            })

    return unique_results
