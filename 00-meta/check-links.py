#!/usr/bin/env python3
"""Link and navigation integrity checker for the ML-overview repo.

Run from anywhere:  python3 00-meta/check-links.py

Checks three things:
  1. Every relative .md link in every file resolves to a real file.
  2. Every SUMMARY.md entry points at a real file  (GitBook nav integrity).
  3. Every .md file on disk appears in SUMMARY.md  (nothing unpublished).

Exit code 0 = clean, 1 = problems found. Safe to run in CI or a pre-commit hook.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".obsidian", ".gitbook", ".claude", "node_modules"}

# [text](target)  — capture the target only
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def md_files():
    for p in sorted(REPO.rglob("*.md")):
        if not any(part in SKIP_DIRS for part in p.relative_to(REPO).parts):
            yield p


def is_internal(target: str) -> bool:
    """Relative link to a markdown file inside the repo."""
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return False
    return ".md" in target


def resolve(src: Path, target: str) -> Path:
    """Resolve a link target relative to the linking file, dropping any #anchor."""
    clean = target.split("#", 1)[0]
    base = REPO if clean.startswith("/") else src.parent
    return (base / clean.lstrip("/")).resolve()


def check_broken_links():
    broken = []
    for f in md_files():
        for target in LINK_RE.findall(f.read_text(encoding="utf-8", errors="ignore")):
            if not is_internal(target):
                continue
            if not resolve(f, target).is_file():
                broken.append((f.relative_to(REPO), target))
    return broken


def check_summary():
    """Return (dead_entries, unlisted_files) for SUMMARY.md."""
    summary = REPO / "SUMMARY.md"
    if not summary.is_file():
        return [], []

    listed, dead = set(), []
    for target in LINK_RE.findall(summary.read_text(encoding="utf-8", errors="ignore")):
        if not is_internal(target):
            continue
        path = resolve(summary, target)
        if path.is_file():
            listed.add(path)
        else:
            dead.append(target)

    unlisted = [
        f.relative_to(REPO)
        for f in md_files()
        if f.resolve() not in listed
        and f.name not in {"SUMMARY.md", "README.md"}
        and "00-meta" not in f.relative_to(REPO).parts
    ]
    return dead, unlisted


def main():
    broken = check_broken_links()
    dead, unlisted = check_summary()

    print(f"{'=' * 60}\nLINK INTEGRITY REPORT\n{'=' * 60}\n")

    print(f"[1] Broken relative links: {len(broken)}")
    for src, target in broken:
        print(f"      {src} -> {target}")

    print(f"\n[2] SUMMARY.md entries pointing at missing files: {len(dead)}")
    for target in dead:
        print(f"      {target}")

    print(f"\n[3] Files on disk absent from SUMMARY.md: {len(unlisted)}")
    for path in unlisted:
        print(f"      {path}")

    total = len(broken) + len(dead) + len(unlisted)
    print(f"\n{'=' * 60}")
    print("CLEAN — no issues found." if total == 0 else f"TOTAL ISSUES: {total}")
    print("=" * 60)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
