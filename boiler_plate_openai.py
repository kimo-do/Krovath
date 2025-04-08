import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import wraps
import json
import logging
import os
from pathlib import Path
import signal
import time
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, or_, select
from openai import AsyncOpenAI, OpenAI, beta


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
def parse_response(response: str, model_class: Type[T]) -> Optional[T]:
    """
    Parses a raw JSON response and validates it against a Pydantic model.
    
    :param response: The raw JSON string.
    :param model_class: The Pydantic model class to validate against.
    :return: An instance of the model if validation succeeds, or None.
    """
    try:
        data = json.loads(response)
        return model_class.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error parsing {model_class.__name__}: {e}")
        return None
    
def extract_and_parse_response(response: object, model_class: Type[T]) -> Optional[T]:
    """
    Extracts the JSON string from an OpenAI response object and parses it into a Pydantic model.
    
    :param response: The response object from your OpenAI API calls.
    :param model_class: The Pydantic model class to validate against.
    :return: An instance of the model if successful, or None.
    """
    json_str = None

    # Look through the response output for the assistant's text message
    if hasattr(response, "output"):
        for item in response.output:
            if getattr(item, "role", None) == "assistant":
                content = item.content
                # content might be a list of messages or a single object
                if isinstance(content, list):
                    for content_item in content:
                        if hasattr(content_item, "text") and content_item.text:
                            json_str = content_item.text
                            break
                elif hasattr(content, "text") and content.text:
                    json_str = content.text
            if json_str:
                break

    if json_str:
        return parse_response(json_str, model_class)
    else:
        print("No valid JSON output found in response.")
        return None
    
def remove_defaults(schema: Any) -> Any:
    """
    Recursively remove any 'default' keys from the schema.
    """
    if isinstance(schema, dict):
        schema.pop("default", None)
        for key, value in schema.items():
            remove_defaults(value)
    elif isinstance(schema, list):
        for item in schema:
            remove_defaults(item)
    return schema

# --- Filter and Tool Building Functions (Your existing code) ---
def create_filter(category: str = None,
                  start_date: datetime = None, end_date: datetime = None,
                  max_days_old: int = None):
    filters = []
    # If max_days_old is provided, calculate date range.
    # Ensure dates are timezone-aware (UTC recommended).
    if max_days_old is not None:
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=max_days_old)
        end_date = now # Use current time as end date

    # Prepare date conditions using Unix timestamps as INTEGERS
    date_conditions = []
    if start_date:
        if start_date.tzinfo is None:
             logger.warning("start_date is naive, assuming UTC for timestamp conversion.")
             start_date = start_date.replace(tzinfo=timezone.utc)
        date_conditions.append({
            "type": "gte",
            "key": "published_at_unix", # Match the metadata key
            "value": int(start_date.timestamp()) # Use integer value for comparison
        })
    if end_date:
        if end_date.tzinfo is None:
             logger.warning("end_date is naive, assuming UTC for timestamp conversion.")
             end_date = end_date.replace(tzinfo=timezone.utc)
        date_conditions.append({
             "type": "lte",
             "key": "published_at_unix", # Match the metadata key
             "value": int(end_date.timestamp()) # Use integer value for comparison
        })

    # ... (rest of the filter combination logic remains the same) ...
    if len(date_conditions) == 2:
         filters.append({
             "type": "and",
             "filters": date_conditions
         })
    elif len(date_conditions) == 1:
        filters.append(date_conditions[0])

    if category is not None:
        filters.append({
            "type": "eq",
            "key": "category",
            "value": category
        })

    if not filters:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        return {
            "type": "and",
            "filters": filters
        }


def build_file_search_tool(vector_store_id: str, category: str = None,
                           start_date: datetime = None, end_date: datetime = None,
                           max_days_old: int = None):
    """
    Build a file search tool definition for the OpenAI API, including filters
    based on metadata (category, date range).
    """
    if not vector_store_id:
         logger.error("Cannot build file search tool without vector_store_id.")
         return None

    # Create the filter structure based on provided parameters
    filters = create_filter(category=category, start_date=start_date, end_date=end_date, max_days_old=max_days_old)
    logger.info(f"Generated filter structure for file search: {filters}")

    # Base tool structure
    tool = {
        "type": "file_search",
        "vector_store_ids": [vector_store_id],  # Use the specific store ID
    }

    # Change key to 'filters' instead of 'filter' per OpenAI API specification
    if filters is not None:
        tool["filters"] = filters  # Assign the generated filter object

    logger.info(f"Built file search tool definition: {tool}")
    return tool


def enforce_required_and_no_additional(schema: Any) -> Any:
    """
    Recursively enforce that each object (with type "object" and "properties") 
    has a 'required' array containing every property key and 'additionalProperties' is false.
    """
    if isinstance(schema, dict):
        # If this is an object with properties, set the required fields and disallow extras.
        if schema.get("type") == "object" and "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
            schema["additionalProperties"] = False
        # Process all values in the dictionary.
        for key, value in schema.items():
            enforce_required_and_no_additional(value)
    elif isinstance(schema, list):
        for item in schema:
            enforce_required_and_no_additional(item)
    return schema

def generate_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    # Generate the initial JSON schema from the model
    schema = model.model_json_schema()
    # Remove any 'default' keys recursively
    schema = remove_defaults(schema)
    # Enforce that every object includes a 'required' array with all keys and no additional properties
    schema = enforce_required_and_no_additional(schema)
    return schema

def log_openai_call(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            total_run_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {total_run_time:.2f} seconds")
            return result
        except TimeoutError as e:
            elapsed_time = time.time() - start_time
            logger.info("OpenAI:", f"{func.__name__} timed out after {elapsed_time:.2f} seconds: {e}")
            return None
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.info("OpenAI:", f"Error in {func.__name__} (after {elapsed_time:.2f}s): {e}")
            return None
    return wrapper