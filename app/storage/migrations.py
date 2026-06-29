import asyncpg

async def run_migrations(pool: asyncpg.Pool):
    async with pool.acquire() as conn:

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Drop old 1536-dim table if exists (new Ollama embedder uses 768)
        await conn.execute("DROP TABLE IF EXISTS embeddings CASCADE;")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id           BIGSERIAL PRIMARY KEY,
                job_id       TEXT NOT NULL,
                sub_question TEXT NOT NULL,
                chunk        TEXT NOT NULL,
                embedding    vector(768),
                created_at   TIMESTAMPTZ DEFAULT now()
            );
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
            ON embeddings
            USING hnsw (embedding vector_cosine_ops);
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS report_store (
                id           BIGSERIAL PRIMARY KEY,
                job_id       TEXT NOT NULL,
                sub_question TEXT NOT NULL,
                summary      TEXT NOT NULL,
                created_at   TIMESTAMPTZ DEFAULT now()
            );
        """)

    print("Migrations complete.")