#!/usr/bin/env python3
"""
Export and upload the hosted Hugging Face Space.

The deploy path defaults to the official Build Small Hackathon org Space. Before
exporting, it seeds the export directory with the live README so judging tags
added on Hugging Face are preserved by scripts/export_hf_space.py.

Usage:
    python scripts/deploy_hf_space.py
    python scripts/deploy_hf_space.py --space-id build-small-hackathon/medical-appt-prep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from export_hf_space import DEFAULT_SPACE_ID, export_space


DEFAULT_EXPORT_DIR = Path("/tmp/medical-appt-prep-hf-space")


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    return subprocess.run(command, check=check, text=True)


def read_json(command: list[str]) -> dict:
    print("+ " + " ".join(command))
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def seed_live_readme(space_id: str, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    result = run(
        [
            "hf",
            "download",
            space_id,
            "README.md",
            "--repo-type",
            "space",
            "--local-dir",
            str(export_dir),
        ],
        check=False,
    )
    if result.returncode != 0:
        print("Warning: could not seed live README tags before export.", file=sys.stderr)


def wait_for_runtime(space_id: str, uploaded_sha: str, timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_info: dict = {}
    while time.monotonic() < deadline:
        last_info = read_json(["hf", "spaces", "info", space_id, "--format", "json"])
        runtime = last_info.get("runtime") or {}
        if runtime.get("sha") == uploaded_sha and runtime.get("stage") == "RUNNING":
            return last_info
        time.sleep(15)
    return last_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", default=DEFAULT_SPACE_ID)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument(
        "--commit-message",
        default="Update Medical Appointment Prep Space",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=600)
    args = parser.parse_args()

    if not args.skip_tests:
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])

    seed_live_readme(args.space_id, args.export_dir)
    export_space(args.export_dir.resolve())

    before = read_json(["hf", "spaces", "info", args.space_id, "--format", "json"])
    run(
        [
            "hf",
            "upload",
            args.space_id,
            str(args.export_dir.resolve()),
            ".",
            "--repo-type",
            "space",
            "--commit-message",
            args.commit_message,
            "--exclude",
            ".cache/**",
            "--exclude",
            "__pycache__/**",
            "--exclude",
            "*.pyc",
            "--exclude",
            "*.zip",
        ]
    )
    after = read_json(["hf", "spaces", "info", args.space_id, "--format", "json"])

    uploaded_sha = after.get("sha")
    final = after
    if uploaded_sha and not args.no_wait:
        final = wait_for_runtime(args.space_id, uploaded_sha, args.wait_timeout)

    print(f"Space: {args.space_id}")
    print(f"Previous SHA: {before.get('sha')}")
    print(f"Uploaded SHA: {uploaded_sha}")
    print(f"Runtime SHA: {(final.get('runtime') or {}).get('sha')}")
    print(f"Runtime stage: {(final.get('runtime') or {}).get('stage')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
