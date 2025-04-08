# db/models.py

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import ARRAY, JSON, TIMESTAMP, BigInteger, Column, Float, ForeignKey, Index, Integer, Numeric, String, DateTime, Boolean, Text, UniqueConstraint, func, text
from sqlalchemy.orm import declarative_base, relationship, validates

# Create the declarative base
Base = declarative_base()

class OpenAIVectorStore(Base):
    __tablename__ = 'openai_vector_store'
    id = Column(String, primary_key=True)  # The vector store id from OpenAI
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CharacterVectorStoreFile(Base):
    __tablename__ = 'character_vector_store_file'
    character_id = Column(Integer, primary_key=True)  # Should match CharacterDB.id
    file_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    vector_store_id = Column(String, index=True, nullable=False)

class CharacterDB(Base):
    __tablename__ = "characters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    character_name = Column(String, nullable=False)
    race = Column(String, nullable=True)
    subrace = Column(String, nullable=True)
    lineage = Column(String, nullable=True)
    region = Column(String, nullable=True)
    unique_feature = Column(String, nullable=True)
    short_character_summary = Column(Text, nullable=True)
    origin_story = Column(Text, nullable=True)
    behavior = Column(Text, nullable=True)
    abilities = Column(Text, nullable=True)
    equipment = Column(Text, nullable=True)
    faction = Column(String, nullable=True)
    cultural_background = Column(Text, nullable=True)
    relationships = Column(ARRAY(String), nullable=True)
    mount = Column(String, nullable=True)
    companion = Column(String, nullable=True)
    positive_traits = Column(ARRAY(String), nullable=True)
    negative_traits = Column(ARRAY(String), nullable=True)
    additional_info = Column(JSON, nullable=True)
    appearance = Column(JSON, nullable=True)
    special_effects = Column(Text, nullable=True)
    visual_description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<CharacterDB(name='{self.character_name}', race='{self.race}', region='{self.region}')>"