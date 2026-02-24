.PHONY: test lint format typecheck clean

test:
	python -m pytest tests/ --cov=platform_abm --cov-report=term-missing -q

lint:
	python -m ruff check platform_abm/ tests/ experiments/

format:
	python -m ruff format platform_abm/ tests/ experiments/
	python -m ruff check --fix platform_abm/ tests/ experiments/

typecheck:
	python -m mypy platform_abm/

clean:
	rm -rf __pycache__ .mypy_cache .ruff_cache .pytest_cache *.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
