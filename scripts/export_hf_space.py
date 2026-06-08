#!/usr/bin/env python3
"""
Export a Hugging Face Space repository from the local app.

Usage:
    python scripts/export_hf_space.py /path/to/hf-space-repo
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPACE_TEMPLATE = ROOT / "deploy" / "huggingface-space"

FILES = [
    "config_loader.py",
]

DIRS = [
    "assets",
    "config",
    "data",
    "src",
]


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_dir(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.zip"),
    )


def export_space(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    copy_file(SPACE_TEMPLATE / "README.md", dest / "README.md")
    copy_file(SPACE_TEMPLATE / "requirements.txt", dest / "requirements.txt")
    copy_file(SPACE_TEMPLATE / "app.py", dest / "app.py")
    copy_file(ROOT / "app.py", dest / "shared_app.py")

    for file_name in FILES:
        copy_file(ROOT / file_name, dest / file_name)
    for dir_name in DIRS:
        copy_dir(ROOT / dir_name, dest / dir_name)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip())
        return 2

    export_space(Path(sys.argv[1]).expanduser().resolve())
    print(f"Exported Hugging Face Space to {Path(sys.argv[1]).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
