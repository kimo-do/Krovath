import asyncio
import io
import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Tuple

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func, delete

# Import AsyncOpenAI from your OpenAI client library
# Make sure your openai library is up-to-date (pip install --upgrade openai)
from openai import AsyncOpenAI
from openai import NotFoundError

# Import our models – note that we now import both the vector store model and the
# character file mapping model (which you should define similarly to VideoVectorStoreFile)
from db.models import CharacterVectorStoreFile, OpenAIVectorStore, CharacterDB

# Set up your logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Existing Setup Code (dotenv, config, SQLAlchemy, OpenAI client) ---
if not os.getenv("RENDER"):
    dotenv_path = os.path.join(os.path.dirname(__file__) if '__file__' in locals() else os.getcwd(), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=True)
        logger.info(f"Loaded .env file: {dotenv_path}")
    else:
        logger.warning(f".env file not found at {dotenv_path}")

# Configuration
DATABASE_URL = os.getenv("DB_PATH", "postgresql+asyncpg://user:password@localhost/dbname")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")

try:
    engine = create_async_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
except Exception as e:
    logger.error(f"Failed to create database engine or session: {e}")

try:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")

# --- FUNCTIONS ADAPTED FOR CHARACTER SHEETS ---


async def upload_korvath_lore_file_to_vector_store(vector_store_id: str) -> Tuple[str, str]:
    """
    Reads the text file at 'files/korvath_lore.txt', checks for an existing file with the same name
    and purpose in OpenAI's file store, and if found, detaches and deletes it.
    Then, uploads the file to OpenAI with purpose "assistants" and attaches it to the specified vector store.
    
    Returns:
        A tuple (file_id, filename) on success.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For any issues during the upload or attachment process.
    """
    # Compute the absolute path to the file relative to this script.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "files", "krovath_lore.txt")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    
    # Check for existing files with the same name and purpose
    try:
        files_list = await client.files.list()
        if hasattr(files_list, "data"):
            for existing_file in files_list.data:
                if existing_file.filename == filename and existing_file.purpose == "assistants":
                    logger.info(f"Found existing file '{filename}' with ID {existing_file.id}. Deleting...")
                    try:
                        # Detach from the vector store first
                        await client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=existing_file.id)
                        logger.info(f"Detached existing file {existing_file.id} from vector store {vector_store_id}.")
                    except Exception as e:
                        logger.warning(f"Could not detach file {existing_file.id} from vector store: {e}")
                    try:
                        await client.files.delete(file_id=existing_file.id)
                        logger.info(f"Deleted existing file {existing_file.id} with filename '{filename}'.")
                    except Exception as e:
                        logger.error(f"Error deleting existing file {existing_file.id}: {e}")
    except Exception as e:
        logger.error(f"Error listing OpenAI files: {e}")

    try:
        # Read file content and prepare for upload
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_obj = io.BytesIO(content.encode("utf-8"))
        
        # Upload the file to OpenAI with purpose "assistants"
        file_response = await client.files.create(file=(filename, file_obj), purpose="assistants")
        file_id = file_response.id
        logger.info(f"Uploaded lore file '{filename}' (OpenAI file_id: {file_id})")

        # Attach the uploaded file to the specified vector store
        vs_file = await client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=file_id)
        logger.info(f"Attached file {file_id} to vector store {vector_store_id}. Status: {vs_file.status}")

        return file_id, filename

    except Exception as e:
        logger.error(f"Error uploading lore file: {e}")
        raise

async def upload_character_json_sheet(character: CharacterDB, vector_store_id: str) -> Tuple[str, str]:
    """
    Converts a character record into JSON format and uploads it as a file to OpenAI's file store.
    The file is then attached to the specified vector store and its metadata is updated.
    
    Returns:
        A tuple (file_id, filename) on successful upload.
    """
    # Build a dictionary from the character record.
    character_data = {
        "character_name": character.character_name,
        "race": character.race,
        "subrace": character.subrace,
        "lineage": character.lineage,
        "region": character.region,
        "unique_feature": character.unique_feature,
        "short_character_summary": character.short_character_summary,
        "origin_story": character.origin_story,
        "behavior": character.behavior,
        "abilities": character.abilities,
        "equipment": character.equipment,
        "faction": character.faction,
        "cultural_background": character.cultural_background,
        "relationships": character.relationships or [],
        "mount": character.mount,
        "companion": character.companion,
        "positive_traits": character.positive_traits or [],
        "negative_traits": character.negative_traits or [],
        "additional_info": character.additional_info or {},
        "appearance": character.appearance or {},
        "special_effects": character.special_effects,
        "visual_description": character.visual_description,
    }
    
    # Convert the dictionary to a formatted JSON string.
    json_str = json.dumps(character_data, indent=2)
    filename = f"character_{character.id}.json"
    file_obj = io.BytesIO(json_str.encode("utf-8"))
    
    try:
        # Upload the JSON file to OpenAI with the purpose "assistants"
        file_response = await client.files.create(file=(filename, file_obj), purpose="assistants")
        file_id = file_response.id
        logger.info(f"Uploaded character JSON sheet for character {character.id} as file '{filename}' (OpenAI file_id: {file_id}).")
        
        # Attach the uploaded file to the specified vector store.
        vs_file = await client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=file_id)
        logger.info(f"Attached file {file_id} to vector store {vector_store_id}. Status: {vs_file.status}.")
        
        # Prepare metadata to update: name, race, generation_date, and region.
        generation_date = int(character.created_at.timestamp()) if character.created_at else int(time.time())
        metadata = {
            "name": character.character_name,
            "race": character.race or "Unknown",
            "generation_date": generation_date,
            "region": character.region or ""
        }
        
        # Update the metadata for the file attached to the vector store.
        await client.vector_stores.files.update(
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=metadata
        )
        logger.info(f"Updated metadata for file {file_id} in vector store {vector_store_id}: {metadata}")
        
        return file_id, filename
    except Exception as e:
        logger.error(f"Error uploading character JSON sheet for character {character.id}: {e}")
        raise


async def get_or_create_vector_store():
    """
    Retrieve the vector store record from the database or create a new one if not found.
    Uses the name "CharacterSheetsStore" for clarity.
    """
    async with SessionLocal() as session:
        result = await session.execute(select(OpenAIVectorStore))
        vector_store_record = result.scalars().first()
        if vector_store_record:
            try:
                # Verify the vector store exists via the API.
                vs = await client.vector_stores.retrieve(vector_store_id=vector_store_record.id)
                logger.info(f"Retrieved existing vector store: {vector_store_record.id}")
                return vector_store_record
            except NotFoundError:
                logger.warning(
                    f"Vector store {vector_store_record.id} found in DB but not in OpenAI. "
                    "Returning stored record."
                )
                return vector_store_record
            except Exception as e:
                logger.error(
                    f"Error verifying vector store {vector_store_record.id}. Returning local record. Error: {e}"
                )
                return vector_store_record

        # Create new vector store via the API if none exists locally
        try:
            vs = await client.vector_stores.create(name="CharacterSheetsStore")
            new_record = OpenAIVectorStore(id=vs.id, name="CharacterSheetsStore")
            session.add(new_record)
            await session.commit()
            logger.info(f"Created new vector store: {vs.id}")
            return new_record
        except Exception as e:
            logger.error(f"Failed to create a new vector store: {e}")
            raise


async def get_vector_store_id():
    vector_store = await get_or_create_vector_store()
    # Return the vector store id if available, otherwise None.
    return vector_store.id if vector_store and hasattr(vector_store, "id") else None

async def fetch_recent_characters():
    """
    Query the database for characters created within the last 7 days.
    Adjust the filtering logic as needed. For example, you might want to only process
    characters that have a non-empty name or have not yet been processed.
    """
    async with SessionLocal() as session:
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        stmt = select(CharacterDB).where(
            CharacterDB.created_at >= seven_days_ago,
            CharacterDB.character_name != ""
        )
        result = await session.execute(stmt)
        characters = result.scalars().all()
    logger.info(f"Fetched {len(characters)} recent characters.")
    return characters

async def get_processed_character_ids():
    """
    Query the database table that maps characters to their OpenAI file IDs.
    Returns a set of character IDs that have already been processed.
    """
    processed = set()
    try:
        async with SessionLocal() as session:
            result = await session.execute(select(CharacterVectorStoreFile.character_id))
            processed = {row[0] for row in result.all()}
    except Exception as e:
        logger.error(f"Failed to query processed character IDs: {e}")
    return processed

async def upload_character_sheet(character: CharacterDB, content: str, vector_store_id: str):
    """
    Prepares and uploads a character sheet file based on the character's data.
    """
    created_date = character.created_at.strftime("%Y-%m-%d") if character.created_at else "Unknown Date"
    # Build file content with key metadata from the character record
    file_content = (
        f"Character Name: {character.character_name}\n"
        f"Race: {character.race}\n"
        f"Subrace: {character.subrace}\n"
        f"Region: {character.region}\n"
        f"Created At: {created_date}\n\n"
        f"{content}"
    )
    file_obj = io.BytesIO(file_content.encode("utf-8"))
    filename = f"character_{character.id}.txt"
    # Upload the file to OpenAI (using purpose "assistants" for vector store usage)
    file_response = await client.files.create(file=(filename, file_obj), purpose="assistants")
    file_id = file_response.id
    logger.info(f"Uploaded character {character.id} sheet as file '{filename}' (OpenAI file_id: {file_id})")
    # Attach the uploaded file to the vector store
    vs_file = await client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=file_id)
    logger.info(f"Attached file {file_id} to vector store {vector_store_id}. Status: {vs_file.status}")
    return file_id, filename

async def update_file_metadata(vector_store_id: str, file_id: str, character: CharacterDB):
    """
    Update metadata for the given file based on the character object.
    Includes fields like character_id, character_name, race, and creation timestamp.
    """
    # Truncate character name if needed
    original_name = str(character.character_name) if character.character_name else ""
    max_len = 254
    truncated_name = original_name[:max_len]
    if len(original_name) > max_len:
        logger.warning(f"Truncated character name for {character.id} to {max_len} chars.")
    attributes_payload = {
        "character_id": str(character.id),
        "character_name": truncated_name,
        "race": character.race or "Unknown",
        "created_at_unix": int(character.created_at.timestamp()) if character.created_at else None,
        "created_at_iso": character.created_at.isoformat() if character.created_at else None,
    }
    # Remove None values
    attributes_payload = {k: v for k, v in attributes_payload.items() if v is not None}
    try:
        await client.vector_stores.files.update(
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=attributes_payload
        )
        logger.info(f"Updated attributes for file {file_id} in vector store {vector_store_id}.")
        logger.debug(f"Attributes set: {attributes_payload}")
    except Exception as e:
        logger.error(f"Failed to update attributes for file {file_id}: {e}")
        raise

async def log_file_metadata(vector_store_id: str, file_id: str) -> None:
    """
    Retrieves metadata for a specific file within a vector store and logs it.
    """
    logger.info(f"Retrieving metadata for file {file_id} from vector store {vector_store_id}...")
    try:
        vs_file_info = await client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=file_id
        )
        logger.info(f"Retrieved file info for {file_id}:")
        logger.info(f"  File ID: {vs_file_info.id}")
        logger.info(f"  Vector Store ID: {vs_file_info.vector_store_id}")
        logger.info(f"  Status: {vs_file_info.status}")
        if vs_file_info.last_error:
            logger.warning(f"  Last Error: Code={vs_file_info.last_error.code}, Message={vs_file_info.last_error.message}")
        if hasattr(vs_file_info, 'metadata') and vs_file_info.metadata:
            logger.info(f"  Metadata: {vs_file_info.metadata}")
        else:
            logger.info("  No metadata associated with this file.")
    except NotFoundError:
        logger.error(f"File {file_id} not found in vector store {vector_store_id}.")
    except Exception as e:
        logger.error(f"Error retrieving metadata for file {file_id}: {e}")

async def cleanup_old_files(vector_store_id):
    """
    Deletes files from the vector store and OpenAI storage that are older than 7 days,
    and removes their corresponding mapping records.
    """
    one_week_ago_unix = time.time() - 7 * 24 * 60 * 60
    deleted_count = 0
    failed_deletions = 0

    try:
        logger.info(f"Starting cleanup of files older than 7 days in vector store {vector_store_id}...")
        store_files = await client.vector_stores.files.list(vector_store_id=vector_store_id)
        for f in store_files.data:
            if f.created_at < one_week_ago_unix:
                file_id = f.id
                logger.info(f"File {file_id} (created at {datetime.fromtimestamp(f.created_at, timezone.utc)}) is older than 7 days. Deleting...")
                try:
                    await client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=file_id)
                    logger.info(f"Deleted file {file_id} from vector store {vector_store_id}.")
                    await client.files.delete(file_id=file_id)
                    logger.info(f"Deleted file {file_id} from OpenAI storage.")
                    async with SessionLocal() as session:
                        delete_stmt = delete(CharacterVectorStoreFile).where(CharacterVectorStoreFile.file_id == file_id)
                        result = await session.execute(delete_stmt)
                        await session.commit()
                        if result.rowcount > 0:
                            logger.info(f"Deleted DB mapping record for file {file_id}.")
                        else:
                            logger.warning(f"No DB mapping record found for file {file_id}.")
                    deleted_count += 1
                except NotFoundError:
                    logger.warning(f"File {file_id} not found during cleanup.")
                    try:
                        async with SessionLocal() as session:
                            delete_stmt = delete(CharacterVectorStoreFile).where(CharacterVectorStoreFile.file_id == file_id)
                            result = await session.execute(delete_stmt)
                            await session.commit()
                            if result.rowcount > 0:
                                logger.info(f"Cleaned up orphaned DB record for file {file_id}.")
                    except Exception as db_e:
                        logger.error(f"Error cleaning DB record for file {file_id}: {db_e}")
                except Exception as e:
                    logger.error(f"Failed to delete file {file_id}: {e}")
                    failed_deletions += 1
        logger.info(f"Cleanup finished for vector store {vector_store_id}. Deleted {deleted_count} files; {failed_deletions} failures.")
    except Exception as e:
        logger.error(f"Cleanup error for vector store {vector_store_id}: {e}")

async def process_character_sheet_concurrently(character: CharacterDB, vector_store_id: str, semaphore: asyncio.Semaphore):
    """
    Processes a single character record:
      1. Uploads the character sheet file.
      2. Polls until file processing is complete.
      3. Updates metadata and logs file info.
      4. Saves the mapping record in the local DB.
    """
    async with semaphore:
        logger.info(f"Processing character {character.id}...")
        # Generate a file content from the character record.
        # Here you can combine various fields; adjust as needed.
        content = (
            f"Subrace: {character.subrace}\n"
            f"Lineage: {character.lineage}\n"
            f"Unique Feature: {character.unique_feature}\n"
            f"Summary: {character.short_character_summary}\n"
            f"Origin: {character.origin_story}\n"
            f"Abilities: {character.abilities}\n"
            f"Equipment: {character.equipment}\n"
            f"Faction: {character.faction}\n"
            f"Cultural Background: {character.cultural_background}\n"
            f"Additional Info: {character.additional_info}\n"
            f"Appearance: {character.appearance}\n"
            f"Special Effects: {character.special_effects}\n"
            f"Visual Description: {character.visual_description}\n"
        )

        try:
            file_id, filename = await upload_character_sheet(character, content, vector_store_id)
        except Exception as e:
            logger.error(f"Failed to upload character sheet for {character.id}: {e}")
            return

        try:
            vs_file_status = await client.vector_stores.files.retrieve(vector_store_id=vector_store_id, file_id=file_id)
            while vs_file_status.status == 'in_progress':
                logger.info(f"File {file_id} still processing. Waiting...")
                await asyncio.sleep(5)
                vs_file_status = await client.vector_stores.files.retrieve(vector_store_id=vector_store_id, file_id=file_id)
            if vs_file_status.status != 'completed':
                logger.error(f"File {file_id} processing failed with status {vs_file_status.status}.")
                return
        except Exception as status_e:
            logger.error(f"Error polling status for file {file_id}: {status_e}")
            return

        try:
            await update_file_metadata(vector_store_id, file_id, character)
            await log_file_metadata(vector_store_id, file_id)
        except Exception as meta_e:
            logger.error(f"Metadata update failed for character {character.id} (file {file_id}): {meta_e}")
            return

        try:
            async with SessionLocal() as session:
                new_record = CharacterVectorStoreFile(
                    character_id=character.id,
                    file_id=file_id,
                    filename=filename,
                    vector_store_id=vector_store_id
                )
                session.add(new_record)
                await session.commit()
            logger.info(f"Completed processing for character {character.id} with file {file_id}.")
        except Exception as db_e:
            logger.error(f"Failed to save mapping for character {character.id}: {db_e}")

async def character_vector_store_main():
    logger.info("Starting character sheet processing workflow...")
    try:
        characters = await fetch_recent_characters()
    except Exception as e:
        logger.error(f"Failed to fetch recent characters: {e}")
        return

    if not characters:
        logger.info("No recent characters found.")
        return

    try:
        vector_store_id = await get_or_create_vector_store()
        if not vector_store_id:
            logger.error("Failed to get or create a vector store. Exiting.")
            return
        logger.info(f"Using vector store: {vector_store_id}")
    except Exception as e:
        logger.error(f"Error with vector store: {e}")
        return

    processed_character_ids = await get_processed_character_ids()
    logger.info(f"Found {len(processed_character_ids)} characters already processed.")

    concurrency_limit = 30  # adjust as needed
    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = []
    for character in characters:
        if character.id in processed_character_ids:
            logger.debug(f"Character {character.id} already processed, skipping.")
            continue
        tasks.append(process_character_sheet_concurrently(character, vector_store_id, semaphore))

    if tasks:
        logger.info(f"Processing {len(tasks)} new characters concurrently...")
        await asyncio.gather(*tasks)
        logger.info("All character sheets processed.")
    else:
        logger.info("No new characters to process.")

    await cleanup_old_files(vector_store_id)
    logger.info("Character sheet processing workflow finished.")

# --- Generic Utility Functions ---
async def delete_all_openai_files():
    """
    Permanently delete all files from your OpenAI account.
    WARNING: This action is irreversible.
    """
    try:
        files = await client.files.list()
        if not files.data:
            print("No files to delete.")
            return
        print(f"Found {len(files.data)} files. Deleting...")
        for f in files.data:
            try:
                await client.files.delete(file_id=f.id)
                print(f"✅ Deleted file {f.id} ({f.filename})")
            except Exception as e:
                print(f"❌ Failed to delete file {f.id}: {e}")
    except Exception as e:
        print(f"Error fetching files: {e}")

async def health_check():
    """
    Performs a health check by:
      - Fetching the list of files from the OpenAI API.
      - Fetching files attached to your vector store.
      - Querying the local database for file mapping records.
    """
    try:
        api_files = await client.files.list()
        api_files_count = len(api_files.data) if hasattr(api_files, "data") else 0
        logger.info(f"Health Check: {api_files_count} files found in OpenAI API.")
    except Exception as e:
        logger.error(f"Health Check: Error fetching files from OpenAI API: {e}")
        api_files_count = None

    try:
        vector_store_id = await get_or_create_vector_store()
        if vector_store_id:
            vector_store_files = await client.vector_stores.files.list(vector_store_id=vector_store_id)
            vector_store_files_count = len(vector_store_files.data)
            logger.info(f"Health Check: {vector_store_files_count} files attached to vector store '{vector_store_id}'.")
        else:
            logger.warning("Health Check: No vector store record found in DB.")
            vector_store_files_count = None
    except Exception as e:
        logger.error(f"Health Check: Error fetching vector store files: {e}")
        vector_store_files_count = None

    try:
        async with SessionLocal() as session:
            result = await session.execute(select(CharacterVectorStoreFile))
            db_files = result.scalars().all()
            db_files_count = len(db_files)
            logger.info(f"Health Check: {db_files_count} file mapping record(s) found in the database.")
            orphaned_files = [f for f in db_files if not f.vector_store_id]
            orphaned_count = len(orphaned_files)
            if orphaned_count:
                logger.warning(f"Health Check: {orphaned_count} orphaned file record(s).")
            else:
                logger.info("Health Check: All file records are associated with a vector store.")
    except Exception as e:
        logger.error(f"Health Check: Error fetching file records from the database: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    run_fresh = "--fresh" in sys.argv
    run_main = not run_fresh

    if run_fresh:
        logger.warning("--- STARTING FRESH START ---")
        logger.warning("This will delete vector stores named 'CharacterSheetsStore' and associated files from OpenAI.")
        async def fresh_start():
            total_deleted_stores = 0
            total_deleted_files = 0
            try:
                logger.info("Listing vector stores to find 'CharacterSheetsStore'...")
                vector_stores = await client.vector_stores.list()
                stores_to_delete = [store for store in vector_stores.data if store.name == "CharacterSheetsStore"]
                if not stores_to_delete:
                    logger.info("No vector stores named 'CharacterSheetsStore' found.")
                else:
                    logger.info(f"Found {len(stores_to_delete)} vector store(s) to delete.")
                    # Process each store sequentially (or concurrently if desired)
                    for store in stores_to_delete:
                        store_id = store.id
                        try:
                            store_files = await client.vector_stores.files.list(vector_store_id=store_id)
                            file_ids = [f.id for f in store_files.data]
                            for file_id in file_ids:
                                try:
                                    await client.vector_stores.files.delete(vector_store_id=store_id, file_id=file_id)
                                    await client.files.delete(file_id=file_id)
                                    async with SessionLocal() as session:
                                        delete_stmt = delete(CharacterVectorStoreFile).where(CharacterVectorStoreFile.file_id == file_id)
                                        await session.execute(delete_stmt)
                                        await session.commit()
                                except Exception as e:
                                    logger.error(f"Error deleting file {file_id}: {e}")
                            await client.vector_stores.delete(vector_store_id=store_id)
                            total_deleted_stores += 1
                        except Exception as e:
                            logger.error(f"Error processing vector store {store_id}: {e}")
            except Exception as e:
                logger.error(f"Error during fresh start: {e}")
            try:
                logger.info("Clearing local helper database tables...")
                async with SessionLocal() as session:
                    del_files_result = await session.execute(delete(CharacterVectorStoreFile))
                    logger.info(f"Deleted {del_files_result.rowcount} records from CharacterVectorStoreFile table.")
                    del_stores_result = await session.execute(delete(OpenAIVectorStore))
                    logger.info(f"Deleted {del_stores_result.rowcount} records from OpenAIVectorStore table.")
                    await session.commit()
            except Exception as e:
                logger.error(f"Error clearing local database tables: {e}")
            logger.info(f"FRESH START COMPLETE: Deleted {total_deleted_stores} vector store(s) and {total_deleted_files} files.")

        logger.info("Executing fresh_start...")
        asyncio.run(fresh_start())
        logger.info("fresh_start finished.")

    if run_main:
        logger.info("Executing main character workflow...")
        # asyncio.run(health_check())
        asyncio.run(character_vector_store_main())
        logger.info("Main workflow finished.")
