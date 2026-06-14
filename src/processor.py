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
    "_The model returned text that could not be formatted into the appointment prep "
    "sections. Please try Generate again._"
)
SECTION_OUTPUT_FALLBACK = (
    "_The model returned text that could not be formatted for this section. "
    "Please try Generate again._"
)

_REASONING_PREFIX_RE = re.compile(
    r"^\s*(?:<think>|<unused\d+>\s*)?(?:thought\b|thinking\b|analysis\b|reasoning\b)",
    flags=re.IGNORECASE,
)
_FINAL_REPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:final(?:\s+answer)?|answer|report)\s*:?\s*(TIMELINE\s*:)",
    flags=re.IGNORECASE,
)
_TIMELINE_RE = re.compile(r"(?:^|\n)\s*TIMELINE\s*:", flags=re.IGNORECASE)


def _strip_reasoning_blocks(text: str) -> str:
    """Remove common hidden-reasoning wrappers before parsing patient output."""
    cleaned = re.sub(r"(?is)<think>.*?</think>\s*", "", text).strip()
    if not _REASONING_PREFIX_RE.search(cleaned):
        timeline_match = _TIMELINE_RE.search(cleaned)
        if timeline_match:
            return cleaned[timeline_match.start() :].strip()
        return cleaned

    final_match = _FINAL_REPORT_RE.search(cleaned)
    if final_match:
        return cleaned[final_match.start(1) :].strip()

    timeline_matches = list(_TIMELINE_RE.finditer(cleaned))
    if len(timeline_matches) > 1:
        return cleaned[timeline_matches[-1].start() :].strip()

    # A reasoning-prefixed response without a clear final report should not be
    # shown to the user as medical prep content.
    return ""


def _limit_list_items(text: str, max_items: int) -> str:
    lines = text.splitlines()
    item_count = 0
    kept: list[str] = []
    for line in lines:
        if re.match(r"\s*(?:[-*]\s+|\d+[.)]\s+)", line):
            item_count += 1
            if item_count > max_items:
                continue
        elif item_count > max_items and not line.strip():
            continue
        elif item_count > max_items:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def clean_section_output(
    raw: str,
    *,
    max_items: int | None = None,
    required_suffix: str = "",
) -> str:
    """Clean one section response without exposing hidden reasoning text."""
    cleaned = parse_output(raw)
    if cleaned.startswith("_No output generated"):
        return cleaned

    cleaned = re.sub(r"(?is)<think>.*?</think>\s*", "", cleaned).strip()
    if _REASONING_PREFIX_RE.search(cleaned):
        final_match = re.search(
            r"(?:^|\n)\s*(?:final(?:\s+answer)?|answer|report)\s*:?\s*(.+)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not final_match:
            return SECTION_OUTPUT_FALLBACK
        cleaned = final_match.group(1).strip()

    cleaned = re.sub(
        r"(?i)^\s*(?:TIMELINE|QUESTIONS|RELEVANT(?:_|\s+)INFO)\s*:\s*",
        "",
        cleaned,
    ).strip()

    cleaned = re.sub(r"(?m)^\s*\*\*[^*\n]{1,80}:?\*\*\s*\n+", "", cleaned).strip()
    first_list_item = re.search(r"(?m)^\s*(?:[-*]\s+|\d+[.)]\s+)", cleaned)
    if first_list_item and first_list_item.start() > 0:
        preamble = cleaned[: first_list_item.start()].strip()
        if len(preamble) <= 240:
            cleaned = cleaned[first_list_item.start() :].strip()

    lines = cleaned.splitlines()
    if len(lines) > 1:
        last_line = lines[-1].strip()
        if last_line and not re.search(r"[.!?*_)]\s*$", last_line):
            cleaned = "\n".join(lines[:-1]).strip()

    if max_items is not None:
        cleaned = _limit_list_items(cleaned, max_items)

    if required_suffix and required_suffix.lower() not in cleaned.lower():
        cleaned = f"{cleaned.rstrip()}\n{required_suffix}".strip()

    return cleaned or SECTION_OUTPUT_FALLBACK


def parse_prep_report(raw: str) -> tuple[str, str, str]:
    """Parse a combined report into timeline, questions, and relevant info."""
    cleaned = _strip_reasoning_blocks(parse_output(raw))
    if cleaned.startswith("_No output generated"):
        return cleaned, cleaned, cleaned
    if not cleaned:
        return REPORT_SECTION_FALLBACK, REPORT_SECTION_FALLBACK, REPORT_SECTION_FALLBACK

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
        return REPORT_SECTION_FALLBACK, REPORT_SECTION_FALLBACK, REPORT_SECTION_FALLBACK

    timeline_match = matches["timeline"]
    questions_match = matches["questions"]
    relevant_match = matches["relevant"]
    assert timeline_match and questions_match and relevant_match

    if not (timeline_match.start() <= questions_match.start() <= relevant_match.start()):
        return REPORT_SECTION_FALLBACK, REPORT_SECTION_FALLBACK, REPORT_SECTION_FALLBACK

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
