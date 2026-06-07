"""
Medication autocomplete data helpers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT_DIR / "data" / "medications" / "rxterms_medications.json"


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
    """Return local RxTerms choices plus common supplement fallback choices."""
    choices: list[str] = []

    if INDEX_PATH.exists():
        with INDEX_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        choices.extend(item["label"] for item in payload.get("medications", []))

    seen = {choice.casefold() for choice in choices}
    for supplement in COMMON_SUPPLEMENTS:
        if supplement.casefold() not in seen:
            choices.append(supplement)
            seen.add(supplement.casefold())

    return choices


def medication_index_summary() -> str:
    if not INDEX_PATH.exists():
        return "Medication autocomplete index is not installed."

    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return (
        f"{payload.get('source', 'RxTerms')} {payload.get('release', '')} "
        f"({payload.get('count', 0):,} medication choices)"
    )
