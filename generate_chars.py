# openai_calls.py
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

from boiler_plate_openai import build_file_search_tool, extract_and_parse_response, generate_schema, log_openai_call
from db.engine import SessionLocal
from db.models import CharacterDB
from file_store import get_vector_store_id, upload_character_json_sheet
from prompts.character_creation import CREATION_PROMPT_INSTRUCTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s: %(message)s"
    # force=True  # requires Python 3.8+
)
logger = logging.getLogger(__name__)

# Adjust this path to where your .env is actually located
if not os.getenv("RENDER"):
    # Load the .env file only if not on Render
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path,override=True)
    print(f"Loaded .env file: {dotenv_path}")

api_key = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=api_key)
ai_model = "o3-mini"

class OpenAiVectorStoreResponse(BaseModel):
    name: str
    race: str
    region: str
    origin_story: str
    unique_trait: str
    behaviour: str
    abilities: str
    additonal_info: str
    visual_description: str
    positive_traits: Optional[List[str]] = None
    negative_traits: Optional[List[str]] = None

@log_openai_call
async def call_vector_store_question(
    query: str,
    model: str = "gpt-4o",
    max_tokens: int = 5000,
) -> Optional[OpenAiVectorStoreResponse]:
    """
    Uses the responses API to query your vector store and parses the result into a structured response.
    """
    # Force the output to match our raw response model (which has 'answer' and 'success' fields)
    response_model = OpenAiVectorStoreResponse

    full_instruction = CREATION_PROMPT_INSTRUCTIONS  # Add any additional instructions if required

    # Get the vector store id (assumes this function is defined elsewhere).
    vector_store_id = await get_vector_store_id()

    # Build the tools list conditionally.
    tools = []
    if vector_store_id:
        file_search_tool = build_file_search_tool(vector_store_id)
        tools.append(file_search_tool)

    # Generate the JSON schema for our response model.
    schema = generate_schema(response_model)

    # Prepare the initial messages.
    input_messages = [{"role": "user", "content": query}]

    # Call responses.create with tools included only if a vector store id was found.
    response = await client.responses.create(
        model=model,
        input=input_messages,
        instructions=full_instruction,
        max_output_tokens=max_tokens,
        temperature=0,
        text={
            "format": {
                "name": response_model.__name__,
                "type": "json_schema",
                "schema": schema,
            }
        },
        tools=tools,
        timeout=120,
    )

    parsed_response = extract_and_parse_response(response, response_model)

    logger.info(parsed_response)
    logger.info("OpenAI: Vector store question returned a structured response.")

    # Generate a nicely formatted string from the parsed response.
    formatted_text = (
        f"Name: {parsed_response.name}\n"
        f"Race: {parsed_response.race}\n"
        f"Region: {parsed_response.region}\n\n"
        "Origin Story:\n"
        f"{parsed_response.origin_story}\n\n"
        f"Unique Trait: {parsed_response.unique_trait}\n\n"
        f"Behaviour: {parsed_response.behaviour}\n\n"
        f"Abilities: {parsed_response.abilities}\n\n"
        f"Additional Info: {parsed_response.additonal_info}\n\n"
        f"Visual Description: {parsed_response.visual_description}\n\n"
    )

    if parsed_response.positive_traits:
        formatted_text += "Positive Traits:\n"
        for trait in parsed_response.positive_traits:
            formatted_text += f"  - {trait}\n"
        formatted_text += "\n"

    if parsed_response.negative_traits:
        formatted_text += "Negative Traits:\n"
        for trait in parsed_response.negative_traits:
            formatted_text += f"  - {trait}\n"
        formatted_text += "\n"

    # Compute the absolute path to character.txt relative to this script.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "character.txt")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        logger.info(f"Character response written to {output_file} with nice formatting.")
    except Exception as e:
        logger.error(f"Failed to write character response to file: {e}")


    return parsed_response




# SECOND PASS


class AdditionalInfo(BaseModel):
    gender: str = ""
    age: str = ""
    element: str = ""
    # Use alias for "class" since it's a reserved word.
    class_: str = Field("", alias="class")
    alignment: str = ""
    beliefs: str = ""
    weapon_type: str = ""
    social_standing: str = ""
    personal_goal: str = ""

class Appearance(BaseModel):
    hair_color: str = ""
    eye_color: str = ""
    build: str = ""
    height: str = ""
    weight: str = ""
    skin_color: str = ""
    clothing_style: str = ""
    armor: str = ""
    gear: str = ""
    notable_traits: str = ""

class CharacterExtractionResponse(BaseModel):
    character_name: str = ""
    race: str = ""
    subrace: str = ""
    lineage: str = ""
    region: str = ""
    unique_feature: str = ""
    short_character_summary: str = ""
    origin_story: str = ""
    behavior: str = ""
    abilities: str = ""
    equipment: str = ""
    faction: str = ""
    cultural_background: str = ""
    relationships: List[str] = []
    mount: str = ""
    companion: str = ""
    positive_traits: List[str] = []
    negative_traits: List[str] = []
    additional_info: AdditionalInfo = AdditionalInfo()
    appearance: Appearance = Appearance()
    special_effects: str = ""
    visual_description: str = ""



@log_openai_call
async def call_character_extraction(
    vector_response: OpenAiVectorStoreResponse,
    model: str = "gpt-4o",
    max_tokens: int = 5000,
) -> Optional[CharacterExtractionResponse]:
    # Prepare a context text based on the vector store response.
    context_text = (
        f"Name: {vector_response.name}\n"
        f"Race: {vector_response.race}\n"
        f"Region: {vector_response.region}\n"
        f"Origin Story: {vector_response.origin_story}\n"
        f"Unique Trait: {vector_response.unique_trait}\n"
        f"Behaviour: {vector_response.behaviour}\n"
        f"Abilities: {vector_response.abilities}\n"
        f"Additional Info: {vector_response.additonal_info}\n"
        f"Visual Description: {vector_response.visual_description}\n"
    )

    logger.info("Call character extraction with context text.")

    # Build the instructions that explain the task.
    instructions = (
        "You are an AI that extracts detailed character information from provided character data. "
        "Below is character data (provided for context, in case it is relevant). "
        "Please extract the information and output valid JSON in the following format. "
        "If any field is not present in the data, leave it empty (i.e. an empty string for text fields or an empty list for arrays):\n\n"
        "{\n"
        '  "character_name": "",\n'
        '  "race": "",\n'
        '  "subrace": "",\n'
        '  "lineage": "",\n'
        '  "region": "",\n'
        '  "unique_feature": "",\n'
        '  "short_character_summary": "",\n'
        '  "origin_story": "",\n'
        '  "behavior": "",\n'
        '  "abilities": "",\n'
        '  "equipment": "",\n'
        '  "faction": "",\n'
        '  "cultural_background": "",\n'
        '  "relationships": [],\n'
        '  "mount": "",\n'
        '  "companion": "",\n'
        '  "positive_traits": [],\n'
        '  "negative_traits": [],\n'
        '  "additional_info": {\n'
        '    "gender": "",\n'
        '    "age": "",\n'
        '    "element": "",\n'
        '    "class": "",\n'
        '    "alignment": "",\n'
        '    "beliefs": "",\n'
        '    "weapon_type": "",\n'
        '    "social_standing": "",\n'
        '    "personal_goal": ""\n'
        "  },\n"
        '  "appearance": {\n'
        '    "hair_color": "",\n'
        '    "eye_color": "",\n'
        '    "build": "",\n'
        '    "height": "",\n'
        '    "weight": "",\n'
        '    "skin_color": "",\n'
        '    "clothing_style": "",\n'
        '    "armor": "",\n'
        '    "gear": "",\n'
        '    "notable_traits": ""\n'
        "  },\n"
        '  "special_effects": "",\n'
        '  "visual_description": ""\n'
        "}\n\n"
        "The following is the character data provided for context:\n"
        f"{context_text}\n"
        "Return only valid JSON."
    )

    # Generate the JSON schema for our response model.
    response_model = CharacterExtractionResponse
    schema = generate_schema(response_model)

    # Build the input content as a single text block.
    input_content = [{"type": "input_text", "text": instructions}]

    response = await client.responses.create(
        model=model,
        instructions=instructions,
        input=[{"role": "user", "content": input_content}],
        max_output_tokens=max_tokens,
        temperature=0,
        text={
            "format": {
                "name": response_model.__name__,
                "type": "json_schema",
                "schema": schema,
            }
        },
        timeout=120,
    )

    parsed_response = extract_and_parse_response(response, response_model)
    logger.info(parsed_response)
    logger.info("OpenAI: Character extraction returned a structured response.")

    # Insert the parsed response into the database.
    try:
        # Assuming you have an async SQLAlchemy session named `session`.
        async with SessionLocal() as session:
            new_character = CharacterDB(
                character_name=parsed_response.character_name,
                race=parsed_response.race,
                subrace=parsed_response.subrace,
                lineage=parsed_response.lineage,
                region=parsed_response.region,
                unique_feature=parsed_response.unique_feature,
                short_character_summary=parsed_response.short_character_summary,
                origin_story=parsed_response.origin_story,
                behavior=parsed_response.behavior,
                abilities=parsed_response.abilities,
                equipment=parsed_response.equipment,
                faction=parsed_response.faction,
                cultural_background=parsed_response.cultural_background,
                relationships=parsed_response.relationships,
                mount=parsed_response.mount,
                companion=parsed_response.companion,
                positive_traits=parsed_response.positive_traits,
                negative_traits=parsed_response.negative_traits,
                additional_info=parsed_response.additional_info.model_dump(),
                appearance=parsed_response.appearance.model_dump(),
                special_effects=parsed_response.special_effects,
                visual_description=parsed_response.visual_description,
            )
            session.add(new_character)
            await session.commit()
            vector_store_id = await get_vector_store_id()
            await upload_character_json_sheet(new_character, vector_store_id)
            logger.info(f"Character '{parsed_response.character_name}' inserted into database.")
    except Exception as e:
        logger.error(f"Failed to insert character into database: {e}")

    return parsed_response

async def main():
    result_str = await call_vector_store_question("Dwarf")
    generated_char = await call_character_extraction(result_str)
    logger.info(generated_char)

if __name__ == '__main__':
    asyncio.run(main())