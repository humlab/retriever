SHELL := /bin/bash
SOURCE_FOLDERS=retriever tests

isort:
	@poetry run isort $(SOURCE_FOLDERS)

black:
	@poetry run black $(SOURCE_FOLDERS)

tidy: isort black
.PHONY: tidy

pylint:
	@poetry run pylint $(SOURCE_FOLDERS)

mypy:
	@poetry run mypy --strict $(SOURCE_FOLDERS)

lint: tidy pylint mypy
.PHONY: lint

clean:
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
.PHONY: clean

test:
	@poetry run pytest tests
.PHONY: test

extract:
	@poetry run python retriever/retriever.py data
.PHONY: extract