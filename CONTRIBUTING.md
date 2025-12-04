# Contributing to Lifegence Voiceprint API

We welcome contributions! This document provides guidelines for contributing.

## Development Setup

### Backend

```bash
# Clone the repository
git clone https://github.com/lifegence/voiceprint-api.git
cd voiceprint-api

# Set up Python environment
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v --cov=app

# Run linting
ruff check app/ tests/
mypy app/
```

### Docker

```bash
# Start all services
docker-compose up -d

# Run tests in Docker
docker-compose --profile test run test
```

## Code Style

### Python

- Use [Black](https://black.readthedocs.io/) for formatting (line-length: 88)
- Use [Ruff](https://docs.astral.sh/ruff/) for linting
- Use [mypy](https://mypy.readthedocs.io/) for type checking

```bash
# Format code
black app/ tests/

# Fix lint issues
ruff check --fix app/ tests/

# Type check
mypy app/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for new functionality
4. Implement your changes
5. Ensure all tests pass: `pytest tests/ -v`
6. Update documentation if needed
7. Submit a pull request

### PR Checklist

- [ ] Tests added for new functionality
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Code formatted (`black app/ tests/`)
- [ ] Linting passes (`ruff check app/ tests/`)
- [ ] Type checking passes (`mypy app/`)
- [ ] Documentation updated

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add speaker group filtering
fix: handle empty audio validation
docs: update API documentation
test: add enrollment edge cases
refactor: simplify embedding extraction
```

## Issue Reporting

When reporting issues, please include:

1. Description of the problem
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (OS, Python version)
5. Relevant logs or error messages

## Security

Report security vulnerabilities privately to **security@lifegence.com**.

Do NOT open public issues for security vulnerabilities.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
