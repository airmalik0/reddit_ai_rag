"""Configuration and constants."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Models
DEFAULT_MODEL = "openai:gpt-5.2"
EMBEDDING_MODEL = "text-embedding-3-small"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"

# Qdrant settings
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_GRPC_PORT = int(os.environ.get("QDRANT_GRPC_PORT", 6334))
COLLECTION_NAME = "reddit_posts"

# Vector store settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Retrieval settings
DEFAULT_NUM_RESULTS = 5
MAX_NUM_RESULTS = 10
