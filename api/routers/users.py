
import json
import logging
import os
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import uuid

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from sqlalchemy import Column, DateTime, Integer, Text, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.auth_verification import verify_bot_token
from db.engine import get_db
from generate_chars import CharacterExtractionResponse, call_character_extraction, call_vector_store_question

router = APIRouter()

# Check if running locally (i.e., not on Render)
if not os.getenv("RENDER"):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    dotenv_path = os.path.join(BASE_DIR, '.env')

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)

# In-memory store for background job statuses and results.
jobs = {}

# Response model returned when scheduling a new job.
class CreateCharacterJobResponse(BaseModel):
    job_id: str
    status: str  # "pending", "completed", or "failed"

# Response model for polling job status.
class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None  # Contains the character data once job is completed.
    error: Optional[str] = None

async def create_character_job(job_id: str):
    """
    Background job that runs the character creation logic:
      1. Calls your vector store question method.
      2. Calls the character extraction which also logs/commits data.
    Updates the 'jobs' dict with the result or error.
    """
    try:
        logger.info(f"Job {job_id}: Starting character creation")
        # You can change the query value ("Dwarf") as needed.
        vector_response = await call_vector_store_question("Dwarf")
        generated_character: CharacterExtractionResponse = await call_character_extraction(vector_response)
        jobs[job_id]["status"] = "completed"
        # Store the result as a dict (use model_dump() if using pydantic v2 or .dict() in v1)
        jobs[job_id]["result"] = generated_character.dict()
        logger.info(f"Job {job_id}: Completed successfully")
    except Exception as e:
        logger.error(f"Job {job_id}: Failed with error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@router.post("/create-character", response_model=CreateCharacterJobResponse)
async def create_character_endpoint(background_tasks: BackgroundTasks):
    """
    Schedules the character creation process in the background.
    Returns a job ID that can be used to poll the status.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "result": None, "error": None}
    background_tasks.add_task(create_character_job, job_id)
    return CreateCharacterJobResponse(job_id=job_id, status="pending")

@router.get("/character-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Polls the status of the background character creation job.
    Returns the current status and, if available, the result or error.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(job_id=job_id, status=job["status"], result=job.get("result"), error=job.get("error"))