# Changelog

All notable changes to this project are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

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
