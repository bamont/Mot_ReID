.PHONY: install lint format test clean

install:
	poetry install --with dev
	poetry run pre-commit install

lint:
	poetry run ruff check src tests
	poetry run black --check src tests

format:
	poetry run ruff check --fix src tests
	poetry run black src tests

test:
	poetry run pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage