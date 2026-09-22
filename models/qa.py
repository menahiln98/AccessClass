"""
Pydantic models for Stage 6: Grounded Learning Agent (retrieval + "Ask This Lecture").
"""

from pydantic import BaseModel, Field


class LectureChunk(BaseModel):
    """One retrievable unit of lecture content — built from one or more Elements."""

    chunk_id: str
    document_id: str
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    text: str = Field(..., min_length=1)


class Citation(BaseModel):
    page_number: int = Field(..., ge=1)
    snippet: str = Field(..., description="Short excerpt of the retrieved chunk that supports the answer.")


class AskAnswer(BaseModel):
    question: str
    answer: str
    is_supported: bool = Field(
        description="False when the lecture material does not contain enough information to answer — "
        "the answer text will say so explicitly rather than inventing a response."
    )
    citations: list[Citation] = Field(default_factory=list)
