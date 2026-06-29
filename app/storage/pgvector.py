import json
from app.storage.db import get_pool

async def store_chunks(
    job_id: str,
    sub_question: str,
    chunks: list[str],
    embeddings: list[list[float]]
):
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO embeddings (job_id, sub_question, chunk, embedding)
            VALUES ($1, $2, $3, $4::vector)
            """,
            [
                (job_id, sub_question, chunk, json.dumps(embedding))
                for chunk, embedding in zip(chunks, embeddings)
            ]
        )

async def retrieve_top_k(
    job_id: str,
    query_embedding: list[float],
    top_k: int = 5
) -> list[str]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT chunk
            FROM embeddings
            WHERE job_id = $1
            ORDER BY embedding <=> $2::vector
            LIMIT $3
            """,
            job_id,
            json.dumps(query_embedding),
            top_k
        )
    return [row["chunk"] for row in rows]