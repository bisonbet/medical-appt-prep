#!/usr/bin/env python3
"""
Build a compact local medication autocomplete index from RxTerms.

RxTerms is derived from RxNorm and published monthly by NLM. The generated JSON
is intentionally checked into the app so local desktop builds and Hugging Face
Spaces can use medication autocomplete without a runtime network dependency.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path


RELEASE = "202606"
RXTERMS_URL = f"https://data.lhncbc.nlm.nih.gov/public/rxterms/release/RxTerms{RELEASE}.zip"
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "medications"
ZIP_PATH = DATA_DIR / f"RxTerms{RELEASE}.zip"
INDEX_PATH = DATA_DIR / "rxterms_medications.json"


def download_release() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        return

    print(f"Downloading {RXTERMS_URL}")
    urllib.request.urlretrieve(RXTERMS_URL, ZIP_PATH)


def main() -> int:
    download_release()

    with zipfile.ZipFile(ZIP_PATH) as archive:
        rxterms_file = next(
            name
            for name in archive.namelist()
            if name.startswith("RxTerms")
            and name.endswith(".txt")
            and "Archive" not in name
            and "Ingredients" not in name
        )

        entries: dict[tuple[str, str], dict[str, str]] = {}
        with archive.open(rxterms_file) as handle:
            rows = (line.decode("utf-8").rstrip("\n") for line in handle)
            reader = csv.DictReader(rows, delimiter="|")
            for row in reader:
                if row.get("SUPPRESS_FOR") or row.get("IS_RETIRED"):
                    continue

                display_name = row.get("DISPLAY_NAME", "").strip()
                strength = row.get("STRENGTH", "").strip()
                if not display_name:
                    continue

                label = f"{display_name} - {strength}" if strength else display_name
                key = (label.casefold(), row.get("RXCUI", ""))
                entries[key] = {
                    "label": label,
                    "rxcui": row.get("RXCUI", "").strip(),
                    "display_name": display_name,
                    "strength": strength,
                    "route": row.get("ROUTE", "").strip(),
                    "dose_form": row.get("NEW_DOSE_FORM", "").strip(),
                    "synonym": row.get("DISPLAY_NAME_SYNONYM", "").strip(),
                }

    medications = sorted(entries.values(), key=lambda item: item["label"].casefold())
    payload = {
        "source": "RxTerms",
        "source_url": RXTERMS_URL,
        "release": RELEASE,
        "generated_on": date.today().isoformat(),
        "count": len(medications),
        "medications": medications,
    }
    INDEX_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(medications)} medications to {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
