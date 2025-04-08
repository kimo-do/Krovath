# db/engine.py

import os
import logging
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Explicitly load the .env file from the parent directory
if not os.getenv("RENDER"):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    dotenv_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

# Retrieve the database URL from environment variables
DB_PATH = os.getenv("DB_PATH")

# Create the asynchronous engine using asyncpg
engine = create_async_engine(DB_PATH, pool_size=10, max_overflow=20, echo=False, pool_pre_ping=True)

# Create the session factory
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    """
    Dependency helper that yields an AsyncSession,
    rolling back on exception and closing afterwards.
    """
    async with SessionLocal() as session:
        try:
            yield session
        except SQLAlchemyError as e:
            await session.rollback()
            logging.error("Database error: %s", e, exc_info=True)
            raise e
        finally:
            await session.close()
