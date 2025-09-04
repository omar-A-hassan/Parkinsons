.PHONY: help install install-dev test test-unit test-integration test-smoke lint format type-check clean preprocess train-wav2vec train-rf

help:
	@echo "Available commands:"
	@echo "  install        Install package in development mode"
	@echo "  install-dev    Install package with development dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration  Run training integration smoke tests"
	@echo "  test-smoke     Alias for test-integration"
	@echo "  lint           Run linting"
	@echo "  format         Format code with black"
	@echo "  type-check     Run type checking with mypy"
	@echo "  clean          Clean build artifacts"
	@echo "  preprocess     Preprocess audio data"
	@echo "  train-wav2vec  Train Wav2Vec2 classifier"
	@echo "  train-rf       Train Random Forest classifier"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,viz,experiment]"

test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v --ignore=tests/test_training_integration.py

test-integration:
	pytest tests/test_training_integration.py -v

test-smoke: test-integration

lint:
	flake8 src tests --max-line-length=100
	black --check src tests

format:
	black src tests

type-check:
	mypy src --ignore-missing-imports

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

preprocess:
	python scripts/preprocess_data.py

train-wav2vec:
	python scripts/train_wav2vec.py

train-rf:
	python scripts/train_random_forest.py

# Setup conda environment
setup-env:
	conda env create -f environment.yml

# Update conda environment
update-env:
	conda env update -f environment.yml --prune