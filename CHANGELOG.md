# Changelog

All notable changes to this project are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## 2.0.0 - 2026-11-05

The project is now a case study in evaluating a regime signal. On real US
equity data, 1963-07-01 to 2026-09-30, the HMM gate lost to the unfiltered book
and to a 21-day vol rule at 5 bps, with 95 % bootstrap intervals below zero on
both books (`docs/REAL_DATA_EVALUATION.md`). By the rule in
`docs/PREREGISTRATION.md`, tuning has stopped.

### Breaking

- Every pinned number changed. The bundled data moved from seed 83 to seed 19,
  and the book now carries a 10 % portfolio vol target (API_SPEC 3.5), so a
  build that was checked against the old `golden.json` will not match.
- `config.json` has a new required key, `portfolio_vol_target`.
- Goldens are 23 cases / 87 scalars. The assertion that crisis momentum is
  negative is gone: on seed 19 it is positive.
- The crisis state is now `argmax` of the variance, not state 0.
  `crisis_momentum_return` reads that state, and code that took state 0 as
  crisis must change.
- `regime_gate` rejects returns <= -100 %.
- The daily portfolio scale adds real turnover: every leg now trades daily,
  including carry between basket changes.

### Skipped by rule

Three conditional experiments were skipped under the stop rule of
`docs/PREREGISTRATION.md`: Student-t emissions, a P(crisis) gate and
real FX carry. Real FX carry being untested also means the combined book has no
real-data result.

### Documentation

- README, LEARN, COOKBOOK, ARCHITECTURE and the site now lead with the
  real-data failure, and give bundled gate figures only beside the statement
  that seed 19 flatters the gate. The four demos print the same caveat.
- LEARN gained sections on why full-sample results mislead, judging a signal,
  detection against profit, turnover and the band, fat tails and `calm_guard`,
  the real-data result, and preregistration; later sections were renumbered.
- COOKBOOK gained recipes for `gate_band`, `transition_floor` and `calm_guard`
  in all four languages, and Python recipes for baselines, the bootstrap and
  the cost model.
- Added five Mermaid diagrams. `tools/check_docs.py` now also checks native
  test counts, diagram embedding, the demo caveat and stale seed-83 figures.
  `tools/check_cookbook.py` now uses the platform's classpath separator.

### Changes in this release

- Golden change: added a pinned portfolio vol target (`portfolio_vol_target = 0.10`, API_SPEC 3.5): one causal scale for the combined book, computed from the unfiltered book and shared by every gate. Added `portfolio_scale` and `scaled_book` in all four languages. The book now runs near 10 % instead of about 4 %, and pays turnover daily.
- Golden change: the bundled data is now seed 19, chosen by a rule fixed in advance (API_SPEC 4): the seed closest to the median unfiltered post-warm-up Sharpe of seeds 1 to 100, median 0.5330. On this seed crisis momentum is positive, so the generator assertion and the sign-case claims are gone, and the highest-variance state is not state 0, so `crisis_momentum_return` now reads the highest-variance state. Goldens are now 23 cases / 87 scalars.
- Replaced the README headline with post-warm-up evidence; added `regime.evaluation`, `tools/evaluate_gate.py` and `docs/GATE_EVALUATION.md`.
- Added the real-data and band evaluations (`tools/evaluate_real_data.py`, `tools/evaluate_band.py`), the opt-in cost model `regime.costs` with `tools/evaluate_capacity.py`, hand-calculated tests and `docs/PREREGISTRATION.md`.
- Added opt-in `gate_band` and `transition_floor`, `RegimeFilter` and filter benchmarks in all four languages; `regime_gate` now rejects returns <= -100 %.
- Added CI for all four languages, `pyproject.toml`; goldens are now 21 cases / 80 scalars.
- Rewrote the site, architecture notes, diagrams and demo takeaway to match the evaluation; added a cookbook recipe for `RegimeFilter` and `tools/check_docs.py`; removed the golden assertion that the gate raises Sharpe.
- Added opt-in `calm_guard` in all four languages: a refit whose lowest-variance state has too little stationary weight is flagged in `guard_trips` and repeated on a clipped window. Off by default, so the 21 existing golden cases are unchanged; goldens are now 22 cases / 83 scalars.
- Known issue: with the guard off, which is the default, one extreme print can still capture the calm state. With it on, the mean gate stays 0.01 to 0.05 above clean (API_SPEC 3.3).

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
