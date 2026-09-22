"""
Pydantic models describing Stage 5: Study-Pack Agent output.
"""

from pydantic import BaseModel, Field


class GlossaryTerm(BaseModel):
    term: str
    definition: str


class GlossaryBatch(BaseModel):
    """What Groq is asked to return for one batch of lecture text."""

    terms: list[GlossaryTerm] = Field(default_factory=list)


class AudioChapter(BaseModel):
    title: str
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    storage_path: str = Field(..., description="Path of the chapter's mp3 in Supabase Storage.")


class StudyPack(BaseModel):
    document_filename: str
    html_storage_path: str = Field(..., description="Path of the accessible HTML pack in Supabase Storage.")
    audio_chapters: list[AudioChapter] = Field(default_factory=list)
    glossary: list[GlossaryTerm] = Field(default_factory=list)
