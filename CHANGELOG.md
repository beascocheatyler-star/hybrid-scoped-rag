# Changelog

## Unreleased — M0: baseline and hygiene

### Fixed
- Frontend markdown/XSS and scroll harnesses failed under Node (`print` is not
  defined), which turned CI red whenever Node was present (2 failed, 265
  passed). The test bundlers now shim `print` for Node, and CI pins Node 20.
- README said the project is MIT; `LICENSE` is GPL-3.0. README now matches
  `LICENSE`.

### Changed
- `DENSE_RESCUE_FLOOR` default raised from 0.25 to 0.36. On the 160-question
  eval gate set, 0.25 rescued 5 off-topic and 0 answerable questions; 0.36
  cuts false accepts from 25.0% to 18.8% with no change in false refusals
  (`scripts/measure_rescue.py`, `docs/scope-tuning.md`). Raising this floor can
  only reduce the set of answered questions.

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
- Three public-domain eval corpora in `examples/public_domain/` (US
  Constitution, Faraday's candle lecture I, Darwin ch. III), from Wikisource.
- Datasets at roadmap size: 125 retrieval questions (30 paraphrases, 15
  non-English), 160 gate questions (80/80), 15 follow-ups, 16 answer sets,
  12 injection cases. `tests/test_eval_datasets.py` checks every label against
  its source text.
- `scripts/measure_rescue.py`: rescue-floor sweep over the eval gate set.
