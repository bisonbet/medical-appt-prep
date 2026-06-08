"""
Input validation and output post-processing.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

MIN_SYMPTOM_LENGTH = 10
MAX_FIELD_LENGTH = 4000


def validate_inputs(
    symptoms: str,
    notes: str = "",
    medications: str = "",
) -> list[str]:
    """
    Return a list of human-readable error strings.
    Empty list means inputs are valid.
    """
    errors: list[str] = []

    if not symptoms or not symptoms.strip():
        errors.append("Please describe your symptoms before generating a report.")
    elif len(symptoms.strip()) < MIN_SYMPTOM_LENGTH:
        errors.append(
            f"Symptom description is too short (minimum {MIN_SYMPTOM_LENGTH} characters). "
            "Please provide more detail."
        )

    for label, value in [("Symptoms", symptoms), ("Notes", notes), ("Medications", medications)]:
        if value and len(value) > MAX_FIELD_LENGTH:
            errors.append(
                f"{label} field exceeds the maximum length of {MAX_FIELD_LENGTH} characters. "
                "Please shorten your input."
            )

    return errors


# ---------------------------------------------------------------------------
# Output post-processing
# ---------------------------------------------------------------------------

def parse_output(raw: str) -> str:
    """
    Light cleanup on raw LLM output:
    - Strip leading/trailing whitespace
    - Remove any accidental repeated blank lines (> 2 in a row)
    - Ensure the output is non-empty with a fallback message
    """
    if not raw or not raw.strip():
        return "_No output generated. Check that the model is running and try again._"

    # Collapse 3+ consecutive blank lines to 2
    cleaned = re.sub(r"\n{3,}", "\n\n", raw.strip())

    return cleaned


REPORT_SECTION_FALLBACK = (
    "_The model returned a report, but this section could not be separated cleanly. "
    "Review the full generated text in the relevant information tab._"
)


def parse_prep_report(raw: str) -> tuple[str, str, str]:
    """Parse a one-call report into timeline, questions, and relevant info."""
    cleaned = parse_output(raw)
    if cleaned.startswith("_No output generated"):
        return cleaned, cleaned, cleaned

    markers = {
        "timeline": r"TIMELINE\s*:",
        "questions": r"QUESTIONS\s*:",
        "relevant": r"RELEVANT(?:_|\s+)INFO\s*:",
    }
    matches = {
        name: re.search(pattern, cleaned, flags=re.IGNORECASE)
        for name, pattern in markers.items()
    }

    if not all(matches.values()):
        return REPORT_SECTION_FALLBACK, REPORT_SECTION_FALLBACK, cleaned

    timeline_match = matches["timeline"]
    questions_match = matches["questions"]
    relevant_match = matches["relevant"]
    assert timeline_match and questions_match and relevant_match

    ordered = sorted(
        [
            ("timeline", timeline_match),
            ("questions", questions_match),
            ("relevant", relevant_match),
        ],
        key=lambda item: item[1].start(),
    )
    positions = {name: (match.end(), None) for name, match in ordered}
    for index, (name, _match) in enumerate(ordered[:-1]):
        next_start = ordered[index + 1][1].start()
        positions[name] = (positions[name][0], next_start)
    last_name = ordered[-1][0]
    positions[last_name] = (positions[last_name][0], len(cleaned))

    sections = {}
    for name, (start, end) in positions.items():
        value = cleaned[start:end].strip()
        sections[name] = value or REPORT_SECTION_FALLBACK

    return sections["timeline"], sections["questions"], sections["relevant"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def truncate_for_context(text: str, max_chars: int = 2000) -> str:
    """Truncate a field to avoid blowing the model's context window."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for context window ...]"
