# Changelog

All notable changes to this project are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

- Replaced the README headline with post-warm-up evidence; added `regime.evaluation`, `tools/evaluate_gate.py` and `docs/GATE_EVALUATION.md`.
- Added opt-in `gate_band` and `transition_floor`, `RegimeFilter` and filter benchmarks in all four languages; `regime_gate` now rejects returns <= -100 %.
- Added CI for all four languages, `pyproject.toml`; goldens are now 21 cases / 80 scalars.
- Added opt-in `calm_guard` in all four languages: a refit whose lowest-variance state has too little stationary weight is flagged in `guard_trips` and repeated on a clipped window. Off by default, so the 21 existing golden cases are unchanged; goldens are now 22 cases / 83 scalars.
- Known issue: with the guard off, which is the default, one extreme print can still capture the calm state. With it on, the mean gate stays 0.02 to 0.04 above clean.

### Added

- GitHub Actions CI: Python, C++ (release and ASan/UBSan), Rust, Java and
  cookbook jobs, plus a banned-word guard.
- `python/pyproject.toml` with package metadata and pytest and ruff settings.
- This changelog.

### Changed

- LEARN.md section 12 is now titled "Knowledge check Q&A"; the docs site
  wording matches.
- Corrected the comment on the trailing return calculation in
  `python/src/regime/strategies.py`.

## v1.1.0 - 2026-09-06

### Changed

- Review pass over all four implementations (Python, C++, Rust, Java):
  correctness fixes and new edge-case tests.
- Regenerated the golden values, expanded the references and documentation,
  and added the docs site.
