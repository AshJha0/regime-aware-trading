#!/usr/bin/env python3
"""Compile and run every fenced code block of COOKBOOK.md against the built libraries.

Each ```python / ```cpp / ```rust / ```java block is extracted into a scratch
directory (default: a temporary directory; override with --scratch), wrapped in
a `main` where the language needs one, compiled against the real library and
executed from the matching language directory (so `../data` resolves).

Prerequisites (from the repository root):

    bash cpp/build.sh                       # cpp/build/libregime.a
    (cd rust && cargo build --release)      # rust/target/release/libregime.rlib
    bash java/build.sh                      # java/out/main

Exit status is non-zero if any snippet fails to compile or run.  Run it as
`python3 tools/check_cookbook.py`; add `--only cpp` (or python/rust/java) to
restrict, `-v` to echo each program's output.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COOKBOOK = ROOT / "COOKBOOK.md"
FENCE = re.compile(r"^```(python|cpp|rust|java)\n(.*?)^```", re.S | re.M)


def extract(text: str) -> list[tuple[str, str]]:
    """Return (language, body) for every fenced block, in document order."""
    return [(m.group(1), m.group(2)) for m in FENCE.finditer(text)]


def split_prefix(body: str, prefix: str) -> tuple[list[str], list[str]]:
    """Separate lines starting with ``prefix`` (includes/imports/uses) from the rest."""
    head, rest = [], []
    for line in body.splitlines():
        (head if line.startswith(prefix) else rest).append(line)
    return head, rest


def wrap_cpp(body: str) -> str:
    includes, rest = split_prefix(body, "#include")
    return "\n".join(includes) + "\n\nint main() {\n" + "\n".join(rest) + "\n    return 0;\n}\n"


def wrap_rust(body: str) -> str:
    # `use` items may span lines (`use regime::{\n ... \n};`): take everything
    # up to the first blank line as the prelude.
    prelude, rest = [], []
    lines = body.splitlines()
    i = 0
    while i < len(lines) and lines[i].strip():
        prelude.append(lines[i])
        i += 1
    rest = lines[i:]
    return "\n".join(prelude) + "\n\nfn main() -> regime::Result<()> {\n" + "\n".join(rest) + "\n    Ok(())\n}\n"


def wrap_java(body: str, cls: str) -> str:
    imports, rest = split_prefix(body, "import ")
    return (
        "\n".join(imports)
        + f"\n\npublic class {cls} {{\n    public static void main(String[] args) throws Exception {{\n"
        + "\n".join(rest)
        + "\n    }\n}\n"
    )


def run(cmd: list[str], cwd: Path, verbose: bool, env: dict | None = None) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    out = proc.stdout + proc.stderr
    if verbose and out.strip():
        print(out.rstrip())
    return proc.returncode == 0, out


def check_one(lang: str, idx: int, body: str, scratch: Path, verbose: bool) -> bool:
    name = f"cookbook_{idx:02d}"
    if lang == "python":
        src = scratch / f"{name}.py"
        src.write_text(body)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "python" / "src"))
        ok, out = run([sys.executable, str(src)], ROOT / "python", verbose, env)
    elif lang == "cpp":
        src = scratch / f"{name}.cpp"
        exe = scratch / name
        src.write_text(wrap_cpp(body))
        lib = ROOT / "cpp" / "build" / "libregime.a"
        ok, out = run(
            ["g++", "-std=c++17", "-Wall", "-Wextra", "-O1", "-I", str(ROOT / "cpp" / "include"),
             str(src), str(lib), "-o", str(exe)],
            ROOT / "cpp", verbose,
        )
        if ok:
            ok, out = run([str(exe)], ROOT / "cpp", verbose)
    elif lang == "rust":
        src = scratch / f"{name}.rs"
        exe = scratch / name
        src.write_text(wrap_rust(body))
        deps = ROOT / "rust" / "target" / "release" / "deps"
        rlib = ROOT / "rust" / "target" / "release" / "libregime.rlib"
        ok, out = run(
            ["rustc", "--edition", "2021", "-O", "-L", str(deps), "--extern", f"regime={rlib}",
             str(src), "-o", str(exe)],
            ROOT / "rust", verbose,
        )
        if ok:
            ok, out = run([str(exe)], ROOT / "rust", verbose)
    elif lang == "java":
        cls = f"Cookbook{idx:02d}"
        src = scratch / f"{cls}.java"
        src.write_text(wrap_java(body, cls))
        classes = ROOT / "java" / "out" / "main"
        ok, out = run(
            ["javac", "-Xlint:all", "-cp", str(classes), "-d", str(scratch), str(src)],
            ROOT / "java", verbose,
        )
        if ok:
            ok, out = run(["java", "-cp", f"{classes}:{scratch}", cls], ROOT / "java", verbose)
    else:  # pragma: no cover
        raise ValueError(lang)
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {lang:<6} block {idx:02d}")
    if not ok and not verbose:
        print(out.rstrip())
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["python", "cpp", "rust", "java"], help="restrict to one language")
    ap.add_argument("--scratch", type=Path, help="scratch directory (default: temporary)")
    ap.add_argument("-v", "--verbose", action="store_true", help="echo program output")
    args = ap.parse_args()

    blocks = extract(COOKBOOK.read_text())
    if not blocks:
        print("no fenced blocks found in COOKBOOK.md")
        return 1
    tmp = None
    scratch = args.scratch
    if scratch is None:
        tmp = tempfile.TemporaryDirectory(prefix="cookbook_")
        scratch = Path(tmp.name)
    scratch.mkdir(parents=True, exist_ok=True)

    failures = 0
    counts: dict[str, int] = {}
    for idx, (lang, body) in enumerate(blocks, start=1):
        if args.only and lang != args.only:
            continue
        counts[lang] = counts.get(lang, 0) + 1
        if not check_one(lang, idx, body, scratch, args.verbose):
            failures += 1
    total = sum(counts.values())
    print(f"{total - failures}/{total} snippets OK " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if tmp is not None:
        tmp.cleanup()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
