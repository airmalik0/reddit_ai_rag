#!/usr/bin/env python3
"""Ingestion script: CSV → Postgres → Qdrant.

Usage:
    python scripts/ingest.py                    # Ingest all CSV files
    python scripts/ingest.py --csv data.csv    # Ingest specific file
    python scripts/ingest.py --force           # Force re-index
"""

import argparse
import csv
import sys
from pathlib import Path
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct

from src.db import (
    get_or_create_subreddit,
    bulk_create_posts,
    bulk_create_chunks,
    get_stats,
    get_cursor,
)
from src.config import (
    EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    QDRANT_HOST,
    QDRANT_PORT,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


def load_csv_file(filepath: str) -> tuple[str, list[dict]]:
    """Load CSV file and extract subreddit name.

    Args:
        filepath: Path to CSV file

    Returns:
        Tuple of (subreddit_name, list of post dicts)
    """
    path = Path(filepath)
    subreddit_name = path.stem  # filename without extension

    posts = []
    seen_urls = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row['url']
            if url not in seen_urls:
                seen_urls.add(url)
                posts.append({
                    'url': url,
                    'title': row['title'],
                    'content': row.get('post_text', '') or ''
                })

    return subreddit_name, posts


def ingest_to_postgres(subreddit_name: str, posts: list[dict]) -> list[int]:
    """Ingest posts to Postgres.

    Args:
        subreddit_name: Name of subreddit
        posts: List of post dictionaries

    Returns:
        List of post IDs
    """
    print(f"  Creating subreddit: {subreddit_name}")
    subreddit_id = get_or_create_subreddit(subreddit_name)

    print(f"  Inserting {len(posts)} posts...")
    post_tuples = [
        (subreddit_id, p['url'], p['title'], p['content'])
        for p in posts
    ]
    post_ids = bulk_create_posts(post_tuples)

    return post_ids


def create_chunks(posts: list[dict], post_ids: list[int]) -> list[tuple[int, int, str, Document]]:
    """Split posts into chunks.

    Args:
        posts: List of post dictionaries
        post_ids: List of corresponding post IDs

    Returns:
        List of (post_id, chunk_index, chunk_content, Document) tuples
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for post, post_id in zip(posts, post_ids):
        # Create full content
        content = f"Title: {post['title']}\n\nPost: {post['content']}"

        # Split into chunks
        doc = Document(page_content=content, metadata={'post_id': post_id})
        split_docs = text_splitter.split_documents([doc])

        for i, split_doc in enumerate(split_docs):
            chunks.append((post_id, i, split_doc.page_content, split_doc))

    return chunks


def index_to_qdrant(
    chunks: list[tuple[int, int, str, Document]],
    dense_embeddings: OpenAIEmbeddings,
    sparse_embeddings: FastEmbedSparse,
    client: QdrantClient,
    collection_name: str
) -> list[str]:
    """Index chunks to Qdrant.

    Args:
        chunks: List of (post_id, chunk_index, content, Document) tuples
        dense_embeddings: Dense embedding model
        sparse_embeddings: Sparse embedding model
        client: Qdrant client
        collection_name: Collection name

    Returns:
        List of Qdrant point IDs
    """
    print(f"  Generating embeddings for {len(chunks)} chunks...")

    # Extract documents and metadata
    documents = [chunk[3] for chunk in chunks]
    contents = [doc.page_content for doc in documents]

    # Generate embeddings
    dense_vectors = dense_embeddings.embed_documents(contents)
    sparse_vectors = list(sparse_embeddings.embed_documents(contents))

    # Create points
    points = []
    point_ids = []

    for i, (post_id, chunk_index, content, doc) in enumerate(chunks):
        point_id = str(uuid4())
        point_ids.append(point_id)

        # Get sparse vector
        sparse_vec = sparse_vectors[i]
        indices = sparse_vec.indices if isinstance(sparse_vec.indices, list) else sparse_vec.indices.tolist()
        values = sparse_vec.values if isinstance(sparse_vec.values, list) else sparse_vec.values.tolist()

        points.append(PointStruct(
            id=point_id,
            vector={
                DENSE_VECTOR_NAME: dense_vectors[i],
                SPARSE_VECTOR_NAME: {
                    "indices": indices,
                    "values": values
                }
            },
            payload={
                "post_id": post_id,
                "chunk_index": chunk_index,
                "page_content": content,
                "metadata": {"post_id": post_id}
            }
        ))

    # Upload to Qdrant in batches
    print(f"  Uploading to Qdrant...")
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"    Uploaded {min(i + batch_size, len(points))}/{len(points)} points")

    return point_ids


def save_chunk_mappings(chunks: list[tuple], point_ids: list[str]):
    """Save chunk to Qdrant point ID mappings in Postgres.

    Args:
        chunks: List of (post_id, chunk_index, content, Document) tuples
        point_ids: List of corresponding Qdrant point IDs
    """
    chunk_tuples = [
        (chunk[0], chunk[1], chunk[2], point_id)
        for chunk, point_id in zip(chunks, point_ids)
    ]
    bulk_create_chunks(chunk_tuples)


def ensure_collection_exists(client: QdrantClient, collection_name: str, dense_size: int, force: bool = False):
    """Ensure Qdrant collection exists with correct config.

    Args:
        client: Qdrant client
        collection_name: Collection name
        dense_size: Size of dense vectors
        force: If True, recreate collection
    """
    if force and client.collection_exists(collection_name):
        print(f"  Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        print(f"  Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=dense_size,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )


def ingest_file(filepath: str, dense_embeddings, sparse_embeddings, client, force: bool = False):
    """Ingest a single CSV file.

    Args:
        filepath: Path to CSV file
        dense_embeddings: Dense embedding model
        sparse_embeddings: Sparse embedding model
        client: Qdrant client
        force: If True, force re-index
    """
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print('='*60)

    # Load CSV
    print("\n[1/5] Loading CSV...")
    subreddit_name, posts = load_csv_file(filepath)
    print(f"  Loaded {len(posts)} posts from r/{subreddit_name}")

    # Ingest to Postgres
    print("\n[2/5] Ingesting to Postgres...")
    post_ids = ingest_to_postgres(subreddit_name, posts)
    print(f"  Created {len(post_ids)} posts in database")

    # Create chunks
    print("\n[3/5] Creating chunks...")
    chunks = create_chunks(posts, post_ids)
    print(f"  Created {len(chunks)} chunks")

    # Ensure collection exists
    print("\n[4/5] Setting up Qdrant...")
    dense_size = len(dense_embeddings.embed_query("test"))
    ensure_collection_exists(client, COLLECTION_NAME, dense_size, force=force)

    # Index to Qdrant
    print("\n[5/5] Indexing to Qdrant...")
    point_ids = index_to_qdrant(
        chunks, dense_embeddings, sparse_embeddings, client, COLLECTION_NAME
    )

    # Save mappings
    print("\n  Saving chunk mappings to Postgres...")
    save_chunk_mappings(chunks, point_ids)

    print(f"\n✅ Completed: {subreddit_name}")
    print(f"   Posts: {len(posts)}, Chunks: {len(chunks)}")


def main():
    parser = argparse.ArgumentParser(description="Ingest CSV data to Postgres and Qdrant")
    parser.add_argument("--csv", type=str, help="Path to specific CSV file")
    parser.add_argument("--force", action="store_true", help="Force re-index (recreate Qdrant collection)")
    args = parser.parse_args()

    print("="*60)
    print("REDDIT RAG INGESTION")
    print("="*60)

    # Initialize embeddings
    print("\nInitializing embeddings...")
    dense_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)
    print(f"  Dense: {EMBEDDING_MODEL}")
    print(f"  Sparse: {SPARSE_EMBEDDING_MODEL}")

    # Initialize Qdrant client
    print("\nConnecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"  Connected to {QDRANT_HOST}:{QDRANT_PORT}")

    # Find CSV files
    if args.csv:
        csv_files = [args.csv]
    else:
        csv_files = list(Path(".").glob("*.csv"))
        print(f"\nFound {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

    # Handle --force: recreate collection once before processing files
    if args.force:
        print("\n[FORCE] Recreating Qdrant collection...")
        dense_size = len(dense_embeddings.embed_query("test"))
        ensure_collection_exists(client, COLLECTION_NAME, dense_size, force=True)

    # Process each file (force=False since we already handled it)
    for csv_file in csv_files:
        ingest_file(str(csv_file), dense_embeddings, sparse_embeddings, client, force=False)

    # Print stats
    print("\n" + "="*60)
    print("FINAL STATS")
    print("="*60)

    stats = get_stats()
    print(f"  Subreddits: {stats['subreddits']}")
    print(f"  Posts: {stats['posts']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Indexed chunks: {stats['indexed_chunks']}")

    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"  Qdrant points: {collection_info.points_count}")

    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
