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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def truncate_for_context(text: str, max_chars: int = 2000) -> str:
    """Truncate a field to avoid blowing the model's context window."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for context window ...]"
