"""
Medication autocomplete data helpers.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT_DIR / "data" / "medications" / "rxterms_medications.json"
COMMON_INDEX_PATH = ROOT_DIR / "data" / "medications" / "rxterms_medications_common.json"
DEFAULT_MEDICATION_SEARCH_LIMIT = 120
_NON_WORD_RE = re.compile(r"[^0-9a-z]+")


COMMON_SUPPLEMENTS = [
    "Vitamin A",
    "Vitamin B12",
    "Vitamin C",
    "Vitamin D",
    "Vitamin D3",
    "Vitamin E",
    "Vitamin K",
    "Calcium",
    "Magnesium",
    "Iron",
    "Zinc",
    "Folic acid",
    "Omega-3 fish oil",
    "Multivitamin",
    "Melatonin",
    "Probiotic",
]


@lru_cache(maxsize=1)
def load_medication_choices() -> list[str]:
    """Return local RxTerms choices (pruned to common medications) plus common supplement fallback choices."""
    choices: list[str] = []

    # Use pruned common medications list if available, otherwise fall back to full list
    index_to_use = COMMON_INDEX_PATH if COMMON_INDEX_PATH.exists() else INDEX_PATH
    
    if index_to_use.exists():
        with index_to_use.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        choices.extend(item["label"] for item in payload.get("medications", []))

    seen = {choice.casefold() for choice in choices}
    for supplement in COMMON_SUPPLEMENTS:
        if supplement.casefold() not in seen:
            choices.append(supplement)
            seen.add(supplement.casefold())

    return choices


def _normalize_search_text(value: str) -> str:
    normalized = _NON_WORD_RE.sub(" ", value.casefold())
    return " ".join(normalized.split())


def _compact_search_text(value: str) -> str:
    return _normalize_search_text(value).replace(" ", "")


def _best_sequence_ratio(query: str, choice: str) -> float:
    search_terms = {_compact_search_text(choice)}
    search_terms.update(
        _compact_search_text(part)
        for part in re.split(r"[\s()/,-]+", choice)
        if len(part) >= 4
    )
    search_terms.discard("")
    if not search_terms:
        return 0.0

    return max(SequenceMatcher(None, query, term).ratio() for term in search_terms)


def _medication_match_score(query: str, choice: str) -> tuple[float, ...] | None:
    normalized_query = _normalize_search_text(query)
    compact_query = normalized_query.replace(" ", "")
    normalized_choice = _normalize_search_text(choice)
    compact_choice = normalized_choice.replace(" ", "")

    if not normalized_query:
        return (0, 0)

    if normalized_choice.startswith(normalized_query):
        return (0, len(normalized_choice))

    words = normalized_choice.split()
    word_prefix_positions = [
        index for index, word in enumerate(words) if word.startswith(normalized_query)
    ]
    if word_prefix_positions:
        return (1, word_prefix_positions[0], len(normalized_choice))

    substring_index = normalized_choice.find(normalized_query)
    if substring_index != -1:
        return (2, substring_index, len(normalized_choice))

    compact_index = compact_choice.find(compact_query)
    if compact_index != -1:
        return (3, compact_index, len(compact_choice))

    sequence_ratio = _best_sequence_ratio(compact_query, choice)
    if sequence_ratio >= 0.72:
        return (4, -sequence_ratio, len(compact_choice))

    return None


def filter_medication_choices(
    query: str,
    choices: list[str],
    limit: int | None = DEFAULT_MEDICATION_SEARCH_LIMIT,
) -> list[str]:
    """Return case-insensitive substring and fuzzy medication matches."""
    if not query or not query.strip():
        return list(choices)

    matches = []
    for index, choice in enumerate(choices):
        score = _medication_match_score(query, choice)
        if score is not None:
            matches.append((score, index, choice))

    matches.sort()
    filtered = [choice for _score, _index, choice in matches]
    return filtered[:limit] if limit is not None else filtered


def medication_index_summary() -> str:
    # Prefer common index if available
    index_to_use = COMMON_INDEX_PATH if COMMON_INDEX_PATH.exists() else INDEX_PATH
    
    if not index_to_use.exists():
        return "You can type any medication, vitamin, or supplement below."

    with index_to_use.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    count = int(payload.get("count", 0))
    if count >= 2000:
        return (
            "Over 2,000 common medications are included. "
            "You can also type any medication, vitamin, or supplement below."
        )
    return (
        "Common medications are included. "
        "You can also type any medication, vitamin, or supplement below."
    )
