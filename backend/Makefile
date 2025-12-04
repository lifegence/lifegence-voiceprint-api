# Makefile for Lifegence VoiceID Backend

.PHONY: install install-dev test test-cov lint format run clean

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

# Run tests in TDD watch mode
test-watch:
	pytest tests/ -v --watch

# Lint code
lint:
	ruff check app/ tests/
	mypy app/

# Format code
format:
	black app/ tests/
	ruff check --fix app/ tests/

# Run development server
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete

# Setup development environment
setup-dev: install-dev
	pre-commit install

# Docker commands
docker-build:
	docker build -t lifegence-voiceid-backend .

docker-run:
	docker run -p 8000:8000 lifegence-voiceid-backend
