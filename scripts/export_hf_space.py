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
DEFAULT_SPACE_ID = "build-small-hackathon/medical-appt-prep"

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


def _split_frontmatter(markdown: str) -> tuple[str | None, str]:
    if not markdown.startswith("---"):
        return None, markdown

    lines = markdown.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, markdown

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])

    return None, markdown


def _extract_frontmatter_tags(markdown: str) -> list[str]:
    frontmatter, _body = _split_frontmatter(markdown)
    if frontmatter is None:
        return []

    lines = frontmatter.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("tags:"):
            continue

        inline_value = stripped.removeprefix("tags:").strip()
        if inline_value.startswith("[") and inline_value.endswith("]"):
            return [
                item.strip().strip("'\"")
                for item in inline_value[1:-1].split(",")
                if item.strip()
            ]
        if inline_value:
            return [inline_value.strip("'\"")]

        tags: list[str] = []
        for next_line in lines[index + 1 :]:
            next_stripped = next_line.strip()
            if not next_line.startswith((" ", "\t")) or not next_stripped.startswith("-"):
                break
            tag = next_stripped.removeprefix("-").strip().strip("'\"")
            if tag:
                tags.append(tag)
        return tags

    return []


def _replace_frontmatter_tags(markdown: str, tags: list[str]) -> str:
    if not tags:
        return markdown

    frontmatter, body = _split_frontmatter(markdown)
    tag_block = "tags:\n" + "\n".join(f"  - {tag}" for tag in tags)

    if frontmatter is None:
        return f"---\n{tag_block}\n---\n\n{markdown}"

    lines = frontmatter.splitlines()
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("tags:"):
            output.append(tag_block)
            replaced = True
            index += 1
            while index < len(lines):
                next_line = lines[index]
                next_stripped = next_line.strip()
                if not next_line.startswith((" ", "\t")) or not next_stripped.startswith("-"):
                    break
                index += 1
            continue

        output.append(lines[index])
        index += 1

    if not replaced:
        output.append(tag_block)

    return "---\n" + "\n".join(output) + "\n---\n" + body


def copy_readme_preserving_tags(src: Path, dest: Path) -> None:
    existing_tags = _extract_frontmatter_tags(dest.read_text()) if dest.exists() else []
    copy_file(src, dest)
    if existing_tags:
        dest.write_text(_replace_frontmatter_tags(dest.read_text(), existing_tags))


def export_space(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dest / ".cache", ignore_errors=True)

    copy_readme_preserving_tags(SPACE_TEMPLATE / "README.md", dest / "README.md")
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
