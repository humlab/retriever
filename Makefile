clean:
	@rm -rf .pytest_cache build dist .eggs *.egg-info
	@rm -rf .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +

coverage:
	@uv run -m pytest --cov=src --cov-report=xml --cov-report=html

format:
	@uv run ruff format src tests

ruff:
	@uv run ruff check --fix --output-format concise src tests

test:
	@uv run -m pytest

typing:
	@uv run mypy --strict src tests || true


create_corpus:
	@uv run src/retriever/create_corpus.py data

.PHONY: clean coverage format ruff test typing create_corpus