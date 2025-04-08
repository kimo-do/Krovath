import asyncio
from db.engine import engine
from db.models import Base

async def init_models():
    async with engine.begin() as conn:
        # This will run the synchronous create_all() inside the async connection
        await conn.run_sync(Base.metadata.create_all)
    print("All tables have been created.")

if __name__ == "__main__":
    asyncio.run(init_models())
