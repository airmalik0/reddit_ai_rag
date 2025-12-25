#!/usr/bin/env python3
"""Reindex all posts from PostgreSQL to Qdrant.

Usage:
    python scripts/reindex.py           # Reindex all posts
    python scripts/reindex.py --force   # Recreate collection first
"""

import argparse
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import FastEmbedSparse
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct

from src.db import get_cursor, bulk_create_chunks
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


def get_all_posts():
    """Get all posts from PostgreSQL with subreddit names."""
    with get_cursor() as cur:
        cur.execute('''
            SELECT p.id, p.title, p.content, s.name as subreddit_name
            FROM posts p
            JOIN subreddits s ON p.subreddit_id = s.id
            ORDER BY p.id
        ''')
        return cur.fetchall()


def clear_chunks():
    """Clear all chunks from PostgreSQL."""
    with get_cursor() as cur:
        cur.execute('DELETE FROM chunks')
        print(f"  Cleared chunks table")


def create_chunks(posts):
    """Split posts into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for post in posts:
        post_id = post['id']
        title = post['title']
        content = post['content'] or ''
        subreddit = post['subreddit_name']

        full_content = f"Subreddit: r/{subreddit}\nTitle: {title}\n\nPost: {content}"

        doc = Document(page_content=full_content, metadata={'post_id': post_id})
        split_docs = text_splitter.split_documents([doc])

        for i, split_doc in enumerate(split_docs):
            chunks.append((post_id, i, split_doc.page_content, split_doc))

    return chunks


def ensure_collection_exists(client, collection_name, dense_size, force=False):
    """Ensure Qdrant collection exists."""
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


def index_to_qdrant(chunks, dense_embeddings, sparse_embeddings, client, collection_name):
    """Index chunks to Qdrant."""
    print(f"  Generating embeddings for {len(chunks)} chunks...")

    documents = [chunk[3] for chunk in chunks]
    contents = [doc.page_content for doc in documents]

    # Generate embeddings in batches with retry
    batch_size = 50  # Smaller batches for stability
    dense_vectors = []
    sparse_vectors = []
    max_retries = 5

    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]

        # Retry loop for network errors
        for attempt in range(max_retries):
            try:
                dense_vectors.extend(dense_embeddings.embed_documents(batch))
                sparse_vectors.extend(list(sparse_embeddings.embed_documents(batch)))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # Exponential backoff
                    print(f"    ⚠️  Error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        print(f"    Embedded {min(i + batch_size, len(contents))}/{len(contents)}")

    # Create points
    points = []
    point_ids = []

    for i, (post_id, chunk_index, content, doc) in enumerate(chunks):
        point_id = str(uuid4())
        point_ids.append(point_id)

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
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"    Uploaded {min(i + batch_size, len(points))}/{len(points)} points")

    return point_ids


def save_chunk_mappings(chunks, point_ids):
    """Save chunk to Qdrant point ID mappings in Postgres."""
    chunk_tuples = [
        (chunk[0], chunk[1], chunk[2], point_id)
        for chunk, point_id in zip(chunks, point_ids)
    ]
    bulk_create_chunks(chunk_tuples)


def main():
    parser = argparse.ArgumentParser(description="Reindex posts from PostgreSQL to Qdrant")
    parser.add_argument("--force", action="store_true", help="Recreate Qdrant collection")
    args = parser.parse_args()

    print("=" * 60)
    print("REINDEX: PostgreSQL → Qdrant")
    print("=" * 60)

    # Initialize embeddings
    print("\n[1/6] Initializing embeddings...")
    dense_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)
    print(f"  Dense: {EMBEDDING_MODEL}")
    print(f"  Sparse: {SPARSE_EMBEDDING_MODEL}")

    # Connect to Qdrant
    print("\n[2/6] Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"  Connected to {QDRANT_HOST}:{QDRANT_PORT}")

    # Get all posts
    print("\n[3/6] Loading posts from PostgreSQL...")
    posts = get_all_posts()
    print(f"  Loaded {len(posts)} posts")

    # Show subreddit breakdown
    subreddit_counts = {}
    for post in posts:
        sub = post['subreddit_name']
        subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
    for sub, count in sorted(subreddit_counts.items()):
        print(f"    r/{sub}: {count} posts")

    # Create chunks
    print("\n[4/6] Creating chunks...")
    chunks = create_chunks(posts)
    print(f"  Created {len(chunks)} chunks")

    # Setup Qdrant collection
    print("\n[5/6] Setting up Qdrant collection...")
    dense_size = len(dense_embeddings.embed_query("test"))
    ensure_collection_exists(client, COLLECTION_NAME, dense_size, force=args.force)

    # Clear old chunks from PostgreSQL
    print("\n  Clearing old chunk mappings...")
    clear_chunks()

    # Index to Qdrant
    print("\n[6/6] Indexing to Qdrant...")
    point_ids = index_to_qdrant(chunks, dense_embeddings, sparse_embeddings, client, COLLECTION_NAME)

    # Save mappings
    print("\n  Saving chunk mappings to PostgreSQL...")
    save_chunk_mappings(chunks, point_ids)

    # Final stats
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Posts: {len(posts)}")
    print(f"  Chunks indexed: {len(chunks)}")

    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"  Qdrant points: {collection_info.points_count}")

    print("\n✅ Reindex complete!")


if __name__ == "__main__":
    main()
