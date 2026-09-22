"""
Pydantic models describing the structured output of Stage 3: Subject Interpreter.

`SubjectTagBatch` is the schema the Groq LLM is asked to fill in directly
(via CrewAI's `output_pydantic`) for one batch of elements. `SubjectTaggedDocument`
is the final, filename-attached result that the rest of the app consumes —
built by combining `SubjectTagBatch` with the document's filename after the
LLM call returns.
"""

from pydantic import BaseModel, Field, field_validator

from config import SUPPORTED_CONTENT_TYPES, SUPPORTED_SUBJECTS


class SubjectTag(BaseModel):
    element_id: str = Field(..., description="Must match an Element.id from the ParsedDocument.")
    subject: str = Field(..., description=f"One of: {SUPPORTED_SUBJECTS}")
    content_type: str = Field(..., description=f"One of: {SUPPORTED_CONTENT_TYPES}")
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("subject")
    @classmethod
    def _check_subject(cls, v: str) -> str:
        if v not in SUPPORTED_SUBJECTS:
            raise ValueError(f"subject must be one of {SUPPORTED_SUBJECTS}, got {v!r}")
        return v

    @field_validator("content_type")
    @classmethod
    def _check_content_type(cls, v: str) -> str:
        if v not in SUPPORTED_CONTENT_TYPES:
            raise ValueError(f"content_type must be one of {SUPPORTED_CONTENT_TYPES}, got {v!r}")
        return v


class SubjectTagBatch(BaseModel):
    """What the LLM is asked to return for one batch of elements."""

    tags: list[SubjectTag] = Field(default_factory=list)


class SubjectTaggedDocument(BaseModel):
    """Final Stage 3 output, attached to a specific document."""

    document_filename: str
    tags: list[SubjectTag] = Field(default_factory=list)
