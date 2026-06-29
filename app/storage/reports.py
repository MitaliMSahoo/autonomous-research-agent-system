from app.storage.db import get_pool

async def save_report(
    job_id: str,
    sub_question: str,
    summary: str
):
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO report_store (job_id, sub_question, summary)
            VALUES ($1, $2, $3)
            """,
            job_id,
            sub_question,
            summary
        )

async def get_report(job_id: str) -> list[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT sub_question, summary, created_at
            FROM report_store
            WHERE job_id = $1
            ORDER BY created_at ASC
            """,
            job_id
        )
    return [dict(r) for r in rows]

async def report_exists(job_id: str) -> bool:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM report_store WHERE job_id = $1 LIMIT 1",
            job_id
        )
    return row is not None