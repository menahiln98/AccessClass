"""
Pydantic models describing the structured output of Stage 2: Accessibility Auditor.

Scope note (Portion 1): the proposal also lists "equation cannot be
interpreted confidently" as a barrier type. Judging that confidence is the
job of the Explanation Agent (Stage 4, Gemini-based), which does not exist
yet in Portion 1. So equation-confidence auditing is intentionally deferred
to Portion 2 rather than faked here — every issue type below is something
Stage 1's output can genuinely support today.
"""

from collections import Counter
from typing import Literal, Optional

from pydantic import BaseModel, Field

IssueType = Literal[
    "scanned_page_no_text",
    "missing_headings",
    "image_no_description",
    "table_no_headers",
    "code_no_language_hint",
    "low_confidence_ocr",
]

Severity = Literal["low", "medium", "high"]


class AccessibilityIssue(BaseModel):
    issue_type: IssueType
    page_number: int = Field(..., ge=1)
    element_id: Optional[str] = None
    description: str
    severity: Severity


class AccessibilityReport(BaseModel):
    document_filename: str
    total_pages: int = Field(..., ge=0)
    issues: list[AccessibilityIssue] = Field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return len(self.issues)

    def issue_counts_by_type(self) -> dict[str, int]:
        return dict(Counter(issue.issue_type for issue in self.issues))
