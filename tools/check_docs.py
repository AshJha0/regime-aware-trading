#!/usr/bin/env python3
"""Fail if counts quoted in the docs drift from the repository.

Checks golden case and scalar counts against data/golden/golden.json, the
snippet count against COOKBOOK.md, the C++, Rust and Java test counts against
the test sources, that the README and the site quote the same test counts,
that every diagram source is embedded in ARCHITECTURE.md, that the four demos
print the same caveat and the README shows it, and that figures from the old
seed 83 appear only where they are labelled historical (LEARN.md section 10).
Run from anywhere: python3 tools/check_docs.py
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
DEMOS = (
    "python/demo.py", "cpp/src/demo.cpp", "rust/src/bin/demo.rs",
    "java/src/main/java/com/quant/regime/Demo.java",
)
CAVEAT = (
    "Seed 19 is favourable to the gate: its post-warm-up Sharpe gain (0.54 -> 0.99)",
    "sits at the 85th percentile of 100 seeds, and its 95% interval includes zero.",
    "On real US equity data, 1963-2026, the gate lost to the unfiltered book and",
    "to a 21-day vol rule, with bootstrap intervals below zero. Tuning has stopped.",
    "See docs/REAL_DATA_EVALUATION.md and docs/GATE_EVALUATION.md.",
)
STALE = ("94.85", "6886.45", "1.80 ", "1.87", "default_rng(83)", "default_rng 83")
NATIVE_TESTS = (
    ("C++", "cpp/tests", "*.cpp", r"^\s*TEST(?:_F|_P)?\(", r"(\d+) tests in one binary"),
    ("Rust", "rust/tests", "*.rs", r"#\[test\]", r"# (\d+) integration tests"),
    ("Java", "java/src/test", "*.java", r"@Test\b", r"\*Test classes \((\d+) tests"),
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
    for lang, folder, glob, pattern, quoted in NATIVE_TESTS:
        n = sum(len(re.findall(pattern, f.read_text(encoding="utf8"), re.M)) for f in (ROOT / folder).rglob(glob))
        errors += [f"README.md: says {v} {lang} tests, sources have {n}"
                   for v in re.findall(quoted, read("README.md")) if int(v) != n]
    architecture = read("docs/ARCHITECTURE.md")
    errors += [f"docs/ARCHITECTURE.md: does not embed {f.name}"
               for f in sorted((ROOT / "docs/diagrams").glob("*.mmd"))
               if f.read_text(encoding="utf8").strip() not in architecture]
    for name in (*DEMOS, "README.md"):
        text = read(name).replace("%%", "%")
        errors += [f"{name}: missing caveat line: {line}" for line in CAVEAT if line not in text]
    for name in DOCS:
        text = read(name)
        if name == "LEARN.md":  # section 10 is the labelled historical example
            text = re.sub(r"^## 10\..*?^## 11\.", "", text, flags=re.S | re.M)
        errors += [f"{name}: stale seed-83 figure {v!r}" for v in STALE if v in text]
    print("\n".join(errors) if errors else f"docs OK: {n_cases} cases, {n_scalars} scalars, {n_snippets} snippets")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
