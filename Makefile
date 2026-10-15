.PHONY: help install lint lint-fix typecheck test test-coverage readiness verify-offline audit secrets ci

PY ?= python

help:
	@echo "install        editable install + CI tool set (no torch; the model is not needed by any test)"
	@echo "lint           ruff check"
	@echo "lint-fix       ruff check --fix"
	@echo "typecheck      mypy (informational: see pyproject [tool.mypy])"
	@echo "test           pytest"
	@echo "test-coverage  pytest with coverage"
	@echo "readiness      scripts/check_submission_readiness.py (video URL is added at submission)"
	@echo "verify-offline an armed tripwire trips and the plug goes OFF with the internet blocked"
	@echo "audit          pip-audit (dependency CVEs)"
	@echo "secrets        gitleaks over the full history"
	@echo "ci             lint + test-coverage + readiness + verify-offline + audit"

install:
	$(PY) -m pip install -r requirements-ci.txt
	$(PY) -m pip install --no-deps -e .

# ── Code quality ────────────────────────────────────────────
lint:
	$(PY) -m ruff check .

lint-fix:
	$(PY) -m ruff check --fix .

typecheck:
	-$(PY) -m mypy jaga --ignore-missing-imports

test:
	$(PY) -m pytest

test-coverage:
	$(PY) -m pytest --cov=jaga --cov-report=term-missing --cov-report=xml

readiness:
	$(PY) scripts/check_submission_readiness.py --allow video

verify-offline:
	$(PY) scripts/verify_offline.py

# ── Security ────────────────────────────────────────────────
audit:
	$(PY) -m pip_audit --skip-editable

secrets:
	gitleaks detect --no-banner --redact --log-opts="--all"

ci: lint test-coverage readiness verify-offline audit
