"""Database operations for Reddit RAG."""

import os
import atexit
from contextlib import contextmanager
from typing import Generator
from dataclasses import dataclass

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag_password@localhost:5432/reddit_rag")


# ---------------------------------------------------------------------------
# Connection Pool (Singleton)
# ---------------------------------------------------------------------------

_connection_pool: pool.SimpleConnectionPool | None = None


def get_pool() -> pool.SimpleConnectionPool:
    """Get or create connection pool (singleton).

    Returns:
        SimpleConnectionPool instance
    """
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL
        )
        # Register cleanup on exit
        atexit.register(_cleanup_pool)
    return _connection_pool


def _cleanup_pool():
    """Cleanup connection pool on exit."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


@dataclass
class Post:
    """Post data class."""
    id: int
    subreddit_id: int
    subreddit_name: str
    url: str
    title: str
    content: str


@dataclass
class Chunk:
    """Chunk data class."""
    id: int
    post_id: int
    chunk_index: int
    content: str
    qdrant_point_id: str | None


def get_connection():
    """Get database connection from pool."""
    return get_pool().getconn()


def release_connection(conn):
    """Return connection to pool."""
    get_pool().putconn(conn)


@contextmanager
def get_cursor(dict_cursor: bool = True) -> Generator:
    """Context manager for database cursor.

    Args:
        dict_cursor: If True, returns RealDictCursor

    Yields:
        Database cursor
    """
    conn = get_connection()
    cursor_factory = RealDictCursor if dict_cursor else None
    try:
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# Subreddit Operations
# ---------------------------------------------------------------------------

def get_or_create_subreddit(name: str) -> int:
    """Get or create subreddit by name.

    Args:
        name: Subreddit name

    Returns:
        Subreddit ID
    """
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO subreddits (name)
            VALUES (%s)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            (name,)
        )
        return cur.fetchone()['id']


# ---------------------------------------------------------------------------
# Post Operations
# ---------------------------------------------------------------------------

def create_post(subreddit_id: int, url: str, title: str, content: str) -> int:
    """Create a new post.

    Args:
        subreddit_id: Subreddit ID
        url: Post URL
        title: Post title
        content: Post content

    Returns:
        Post ID
    """
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO posts (subreddit_id, url, title, content)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content
            RETURNING id
            """,
            (subreddit_id, url, title, content)
        )
        return cur.fetchone()['id']


def bulk_create_posts(posts: list[tuple]) -> list[int]:
    """Bulk create posts.

    Args:
        posts: List of (subreddit_id, url, title, content) tuples

    Returns:
        List of post IDs
    """
    with get_cursor() as cur:
        result = execute_values(
            cur,
            """
            INSERT INTO posts (subreddit_id, url, title, content)
            VALUES %s
            ON CONFLICT (url) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content
            RETURNING id
            """,
            posts,
            fetch=True
        )
        return [row['id'] for row in result]


def get_post_by_id(post_id: int) -> Post | None:
    """Get post by ID.

    Args:
        post_id: Post ID

    Returns:
        Post object or None
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.subreddit_id, s.name as subreddit_name,
                   p.url, p.title, p.content
            FROM posts p
            JOIN subreddits s ON p.subreddit_id = s.id
            WHERE p.id = %s
            """,
            (post_id,)
        )
        row = cur.fetchone()
        if row:
            return Post(**row)
        return None


def get_posts_by_ids(post_ids: list[int]) -> list[Post]:
    """Get multiple posts by IDs.

    Args:
        post_ids: List of post IDs

    Returns:
        List of Post objects
    """
    if not post_ids:
        return []

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.subreddit_id, s.name as subreddit_name,
                   p.url, p.title, p.content
            FROM posts p
            JOIN subreddits s ON p.subreddit_id = s.id
            WHERE p.id = ANY(%s)
            """,
            (post_ids,)
        )
        return [Post(**row) for row in cur.fetchall()]


def get_all_posts(subreddit_name: str | None = None) -> list[Post]:
    """Get all posts, optionally filtered by subreddit.

    Args:
        subreddit_name: Optional subreddit name filter

    Returns:
        List of Post objects
    """
    with get_cursor() as cur:
        if subreddit_name:
            cur.execute(
                """
                SELECT p.id, p.subreddit_id, s.name as subreddit_name,
                       p.url, p.title, p.content
                FROM posts p
                JOIN subreddits s ON p.subreddit_id = s.id
                WHERE s.name = %s
                ORDER BY p.id
                """,
                (subreddit_name,)
            )
        else:
            cur.execute(
                """
                SELECT p.id, p.subreddit_id, s.name as subreddit_name,
                       p.url, p.title, p.content
                FROM posts p
                JOIN subreddits s ON p.subreddit_id = s.id
                ORDER BY p.id
                """
            )
        return [Post(**row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Chunk Operations
# ---------------------------------------------------------------------------

def create_chunk(post_id: int, chunk_index: int, content: str, qdrant_point_id: str | None = None) -> int:
    """Create a chunk.

    Args:
        post_id: Post ID
        chunk_index: Index of chunk within post
        content: Chunk content
        qdrant_point_id: Optional Qdrant point ID

    Returns:
        Chunk ID
    """
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (post_id, chunk_index, content, qdrant_point_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (post_id, chunk_index) DO UPDATE SET
                content = EXCLUDED.content,
                qdrant_point_id = EXCLUDED.qdrant_point_id
            RETURNING id
            """,
            (post_id, chunk_index, content, qdrant_point_id)
        )
        return cur.fetchone()['id']


def bulk_create_chunks(chunks: list[tuple]) -> list[int]:
    """Bulk create chunks.

    Args:
        chunks: List of (post_id, chunk_index, content, qdrant_point_id) tuples

    Returns:
        List of chunk IDs
    """
    with get_cursor() as cur:
        result = execute_values(
            cur,
            """
            INSERT INTO chunks (post_id, chunk_index, content, qdrant_point_id)
            VALUES %s
            ON CONFLICT (post_id, chunk_index) DO UPDATE SET
                content = EXCLUDED.content,
                qdrant_point_id = EXCLUDED.qdrant_point_id
            RETURNING id
            """,
            chunks,
            fetch=True
        )
        return [row['id'] for row in result]


def get_chunk_by_qdrant_id(qdrant_point_id: str) -> Chunk | None:
    """Get chunk by Qdrant point ID.

    Args:
        qdrant_point_id: Qdrant point ID

    Returns:
        Chunk object or None
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, post_id, chunk_index, content, qdrant_point_id
            FROM chunks
            WHERE qdrant_point_id = %s
            """,
            (qdrant_point_id,)
        )
        row = cur.fetchone()
        if row:
            return Chunk(**row)
        return None


def get_posts_by_chunk_ids(chunk_ids: list[int]) -> list[Post]:
    """Get posts by chunk IDs.

    Args:
        chunk_ids: List of chunk IDs

    Returns:
        List of unique Post objects
    """
    if not chunk_ids:
        return []

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT p.id, p.subreddit_id, s.name as subreddit_name,
                   p.url, p.title, p.content
            FROM posts p
            JOIN subreddits s ON p.subreddit_id = s.id
            JOIN chunks c ON c.post_id = p.id
            WHERE c.id = ANY(%s)
            """,
            (chunk_ids,)
        )
        return [Post(**row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_stats() -> dict:
    """Get database statistics.

    Returns:
        Dictionary with counts
    """
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) as count FROM subreddits")
        subreddits = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM posts")
        posts = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM chunks")
        chunks = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM chunks WHERE qdrant_point_id IS NOT NULL")
        indexed_chunks = cur.fetchone()['count']

        return {
            'subreddits': subreddits,
            'posts': posts,
            'chunks': chunks,
            'indexed_chunks': indexed_chunks
        }
