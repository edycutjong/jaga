# Contributing

Thanks for your interest in improving jaga.

## Getting started
1. Fork the repo and branch from `main`: `git checkout -b feat/your-change`
2. Install the test tool set (no model download, no GPU, no camera): `make install`
3. Run the checks: `make ci`
4. To run the real thing (Cosmos-Embed1 on your laptop), follow *Getting Started* in the README.

## Before you open a PR
- `make ci` passes: ruff, pytest with coverage, the readiness script, the offline-trip check and pip-audit.
- **The safety rules are not negotiable.** A trip may only switch a plug OFF or notify. No code path reachable from a
  trip may call ON. `tests/test_invariants.py` pins this; keep it green and extend it, never loosen it.
- Add or update tests for any behaviour change, and name a regression test after the defect it pins.
- The readiness check marks a measured section of `DEMO.md` stale when the code it measured changed (the scoring and
  live-loop files). That needs `python scripts/bench.py` re-run on a machine with the model; say so in the PR if it applies.
- If you change the README's test count, it must equal what `pytest` collects (the readiness script checks this).
- Never commit video frames, device keys (`devices.json`, `tinytuya.json`) or `.env` files.
- Keep commits small, with a message that says what changed and why.

## Reporting bugs / requesting features
Open an issue with the templates provided. Include your OS, Python version, the camera and plug you use, and
whether the model ran on MPS, CPU or CUDA.
