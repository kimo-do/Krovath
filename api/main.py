# api/main.py
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import sys
import os

import json
from statistics import mean

from fastapi import Depends, FastAPI, HTTPException, Header, Request, status
import requests
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from base64 import b64decode
from starlette.responses import Response

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import TIMESTAMP, Column, String, DateTime, cast, select, func, Interval, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel
from typing import List, Optional

from db.engine import engine, SessionLocal
from db.models import Base

from api.routers import users

SERVER_URL = "http://localhost:3010"

# Check if running locally (i.e., not on Render)
if not os.getenv("RENDER"):
    # Always load the .env file from the backend root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dotenv_path = os.path.join(BASE_DIR, '.env')
    load_dotenv(dotenv_path)


DB_PATH = os.getenv("DB_PATH")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")
GOOGLE_SEARCH_ID = os.getenv("GOOGLE_CSE_ID")

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URL,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory nonce storage for wallets
nonces = {}

@app.options("/{path:path}")
async def preflight_handler():
    return Response(status_code=200)

@app.get("/")
async def read_root():
    return {"message": "Service is up and running!"}


# Include the video router
app.include_router(users.router)


