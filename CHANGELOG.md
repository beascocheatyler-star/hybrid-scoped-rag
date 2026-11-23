# Changelog

## Unreleased — M0: baseline and hygiene

### Fixed
- Frontend markdown/XSS and scroll harnesses failed under Node (`print` is not
  defined), which turned CI red whenever Node was present (2 failed, 265
  passed). The test bundlers now shim `print` for Node, and CI pins Node 20.
- README said the project is MIT; `LICENSE` is GPL-3.0. README now matches
  `LICENSE`.

### Added
- `tests/test_invariants.py`: pins scope-gate invariants I1–I9 (filter before
  score, isolation, filtered coverage, unknown keys exclude, zero-token
  refusals, monotonic floor, injection handling, empty-scope vs no-corpus,
  key-free `/api/search`).
- `tests/fakes.py`: `FakeLLM`, a deterministic client that records calls.
- `eval/`: offline/live evaluation harness (`python -m eval.run --offline`),
  span-labelled datasets (retrieval, gate, follow-ups, answers, injection),
  metrics, and markdown report. Baseline in `eval/baseline.json` and
  `docs/eval-baseline.md`.
- CI step: offline eval fails the build on a >2-point regression or any
  injection-resistance failure.
- `docs/design/M0.md`.
