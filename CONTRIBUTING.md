# Contributing to EdgeFirst CameraAdaptor

Thank you for your interest in contributing to EdgeFirst CameraAdaptor!

## Code of Conduct

This project follows our [Code of Conduct](./CODE_OF_CONDUCT.md). By participating,
you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

1. Check existing issues first to avoid duplicates
2. Use the bug report template
3. Include reproduction steps and environment details

### Suggesting Features

1. Check existing feature requests
2. Use the feature request template
3. Explain the use case and benefits

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/description`
3. Make changes following our coding standards
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Sign your commits with DCO: `git commit -s`
7. Push and create a pull request

### Commit Messages

Format: `Brief description in imperative mood`

- Use imperative mood ("Add feature" not "Added feature")
- Keep first line under 72 characters
- Sign all commits with DCO: `git commit -s`

Example:
```
Add YUYV to NV12 color space conversion

Signed-off-by: Your Name <your.email@example.com>
```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/EdgeFirstAI/cameraadaptor.git
cd cameraadaptor

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[all,dev]"

# Run tests
pytest

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

## Coding Standards

- Follow PEP-8 with 79 character line limit
- Use type hints for all public APIs
- Write docstrings for modules, classes, and functions
- Maintain test coverage above 70%

## Testing

- Write tests for all new functionality
- Place tests in the `tests/` directory
- Use pytest fixtures from `conftest.py`
- Run the full test suite before submitting PRs

```bash
# Run all tests with coverage
pytest --cov=edgefirst.cameraadaptor --cov-report=term-missing

# Run specific test file
pytest tests/test_color_spaces.py

# Run tests matching a pattern
pytest -k "test_yuyv"
```

## License

By contributing, you agree that your contributions will be licensed under
the project's [Apache-2.0 License](./LICENSE).
