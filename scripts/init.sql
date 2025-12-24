-- Reddit RAG Database Schema

-- Subreddits table
CREATE TABLE IF NOT EXISTS subreddits (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Posts table (source of truth)
CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    subreddit_id INTEGER REFERENCES subreddits(id) ON DELETE CASCADE,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table (for tracking what's indexed in Qdrant)
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    qdrant_point_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(post_id, chunk_index)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit_id);
CREATE INDEX IF NOT EXISTS idx_posts_url ON posts(url);
CREATE INDEX IF NOT EXISTS idx_chunks_post ON chunks(post_id);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_id ON chunks(qdrant_point_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for posts updated_at
DROP TRIGGER IF EXISTS update_posts_updated_at ON posts;
CREATE TRIGGER update_posts_updated_at
    BEFORE UPDATE ON posts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
