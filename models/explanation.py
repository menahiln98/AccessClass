"""
Pydantic models describing Stage 4: Explanation Agent output.

Two raw LLM-facing schemas exist because two different models produce them:
- VisualExplanation is filled in directly by Gemini (images).
- ExplanationBatch/ExplanationItem is filled in directly by Groq, via CrewAI,
  for code/table/equation elements (all text-based, no image needed).

ElementExplanation is the unified, final per-element record the rest of the
app consumes — built by combining whichever raw schema applies.
"""

from typing import Optional

from pydantic import BaseModel, Field


class VisualExplanation(BaseModel):
    """What Gemini is asked to return directly for one image element."""

    short_alt_text: str = Field(..., description="A few words, for quick screen-reader navigation.")
    detailed_explanation: str = Field(..., description="Full context a student needs to understand the image.")
    academic_meaning: str = Field(..., description="Why this visual matters in the lecture.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_unclear: bool = Field(
        default=False,
        description="True if the image is too unclear/low-quality to describe safely.",
    )


class ExplanationItem(BaseModel):
    """What Groq is asked to return for one code/table/equation element."""

    element_id: str
    explanation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    looks_like_real_table: Optional[bool] = Field(
        default=None,
        description="Only set when element_type == 'table': whether the cell content looks like "
        "genuine tabular data rather than a design layout PyMuPDF mistook for a table.",
    )


class ExplanationBatch(BaseModel):
    """What the LLM returns for one batch of code/table/equation elements."""

    items: list[ExplanationItem] = Field(default_factory=list)


class ElementExplanation(BaseModel):
    """Final, unified Stage 4 record for one element — what the rest of the app reads."""

    element_id: str
    element_type: str
    explanation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_review: bool = Field(
        description="True if confidence is low, the image was flagged unclear, or a 'table' "
        "turned out not to look like real tabular data."
    )
    source_model: str


class ExplanationDocument(BaseModel):
    document_filename: str
    explanations: list[ElementExplanation] = Field(default_factory=list)
