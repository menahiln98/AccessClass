import pytest
from pydantic import ValidationError

from models.subject import SubjectTag


def test_valid_subject_tag():
    tag = SubjectTag(element_id="p1-e1", subject="calculus", content_type="equation", confidence=0.9)
    assert tag.subject == "calculus"


def test_invalid_subject_rejected():
    with pytest.raises(ValidationError):
        SubjectTag(element_id="p1-e1", subject="physics", content_type="equation", confidence=0.9)


def test_invalid_content_type_rejected():
    with pytest.raises(ValidationError):
        SubjectTag(element_id="p1-e1", subject="calculus", content_type="video", confidence=0.9)


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SubjectTag(element_id="p1-e1", subject="calculus", content_type="equation", confidence=1.5)


@pytest.mark.parametrize(
    "subject",
    ["ict", "oop", "database_systems", "operating_systems", "computer_networks", "artificial_intelligence_ml"],
)
def test_new_subjects_are_accepted(subject):
    tag = SubjectTag(element_id="p1-e1", subject=subject, content_type="text", confidence=0.9)
    assert tag.subject == subject