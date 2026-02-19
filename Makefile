# AgentSpine — common development commands
# On Windows without make: use the commands under each target (e.g. python -m pytest)

.PHONY: install test lint format check cov clean help

help:
	@echo "install  - Install package with dev deps (pip install -e .[dev])"
	@echo "test     - Run tests (pytest)"
	@echo "cov      - Run tests with coverage report"
	@echo "lint     - Run ruff check"
	@echo "format   - Run ruff format"
	@echo "check    - Lint + format check (CI)"
	@echo "clean    - Remove build artifacts and caches"

install:
	pip install -e ".[dev]"

test:
	pytest

cov:
	pytest --cov=src --cov=main --cov-report=term-missing --no-cov-on-fail

lint:
	ruff check src main.py tests

format:
	ruff format src main.py tests

check: lint
	ruff format --check src main.py tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
