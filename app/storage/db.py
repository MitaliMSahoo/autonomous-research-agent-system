import asyncpg
import os

_pool = None

async def init_pool() -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))
    return _pool

def get_pool() -> asyncpg.Pool:
    
    if _pool is None:
        raise RuntimeError("Pool not initialized. Call init_pool() first.")
    return _pool

async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None