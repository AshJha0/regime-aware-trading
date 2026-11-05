#!/usr/bin/env python3
"""Fail if counts quoted in the docs drift from the repository.

Checks golden case and scalar counts against data/golden/golden.json, the
snippet count against COOKBOOK.md, and that the README and the site quote
the same test counts.  Run from anywhere: python3 tools/check_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    "README.md", "LEARN.md", "COOKBOOK.md", "API_SPEC.md", "docs/ARCHITECTURE.md",
    "docs/index.html", "docs/diagrams/pipeline_flow.mmd",
)
CASES = re.compile(r"(\d+) (?:named |golden |cross-language )*cases\b")
SCALARS = re.compile(r"(\d+) (?:golden )?scalars\b")
SNIPPETS = re.compile(r"(\d+)(?: compiled-and-run| snippets)")
TESTS = re.compile(r"\((?:GoogleTest, )?(\d+) tests|# (\d+) integration tests")


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf8")


def main() -> int:
    cases = json.loads(read("data/golden/golden.json"))["cases"]
    n_cases, n_scalars = len(cases), sum(len(c["expect"]) for c in cases)
    n_snippets = len(re.findall(r"^```(?:python|cpp|rust|java)\r?$", read("COOKBOOK.md"), re.M))
    errors = []
    for name in DOCS:
        text = re.sub(r"\s+", " ", read(name))
        errors += [f"{name}: says {v} cases, golden.json has {n_cases}"
                   for v in CASES.findall(text) if int(v) != n_cases and "golden" in text]
        errors += [f"{name}: says {v} scalars, golden.json has {n_scalars}"
                   for v in SCALARS.findall(text) if int(v) != n_scalars]
    for name in ("README.md", "docs/index.html"):
        errors += [f"{name}: says {v} snippets, COOKBOOK.md has {n_snippets}"
                   for v in SNIPPETS.findall(read(name)) if int(v) != n_snippets]
    readme, site = (sorted(a or b for a, b in TESTS.findall(read(n))) for n in ("README.md", "docs/index.html"))
    if readme != site:
        errors.append(f"test counts differ: README {readme}, site {site}")
    print("\n".join(errors) if errors else f"docs OK: {n_cases} cases, {n_scalars} scalars, {n_snippets} snippets")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
