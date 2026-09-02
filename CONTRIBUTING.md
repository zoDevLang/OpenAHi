# Contributing to OpenAHI

Thank you for your interest in contributing to OpenAHI! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

There are many ways to contribute to OpenAHI:

- **Reporting bugs** - Open issues for bugs you find
- **Suggesting features** - Open issues for feature requests
- **Fixing bugs** - Submit pull requests with bug fixes
- **Implementing features** - Submit pull requests with new features
- **Improving documentation** - Help improve the docs
- **Testing** - Help test the project and report issues

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your changes
4. **Make your changes**
5. **Test your changes**
6. **Commit your changes** with descriptive commit messages
7. **Push to your fork**
8. **Open a pull request** to the main repository

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/zoDevLang/OpenAHi.git
cd OpenAHi

# Install Python dependencies in development mode
pip install -e ./python[dev]

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all Python tests
cd python
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=openahi --cov-report=html

# Run specific test file
python -m pytest tests/test_model.py

# Run specific test
python -m pytest tests/test_model.py::TestComposterModel::test_forward_pass
```

### Building Rust Components

```bash
# Build all Rust components
cd rust
cargo build

# Build in release mode
cargo build --release

# Run Rust tests
cargo test
```

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8 style guide
  - Use `black` for code formatting
  - Use `isort` for import sorting
  - Use `flake8` for linting

- **Rust**: Follow Rust style guidelines
  - Use `rustfmt` for code formatting
  - Use `clippy` for linting

- **C++**: Follow modern C++ best practices
  - Use consistent indentation (4 spaces)
  - Use descriptive variable names
  - Add comments for non-obvious code

### Commit Messages

- Use clear, descriptive commit messages
- Follow the [Conventional Commits](https://www.conventionalcommits.org/) convention
- Include the issue number if applicable

Examples:
- `feat: add tokenizer save/load functionality`
- `fix: correct attention mask in transformer block`
- `docs: update README with installation instructions`
- `refactor: simplify model loading code`

### Pull Request Guidelines

1. **Title**: Clear and descriptive
2. **Description**: Explain what the PR does and why
3. **Linked Issues**: Reference any related issues
4. **Tests**: Include tests for new functionality
5. **Documentation**: Update documentation if needed
6. **Breaking Changes**: Note any breaking changes

### Branch Naming

- Use descriptive branch names
- Use hyphens to separate words
- Include issue number if applicable

Examples:
- `feat/add-tokenizer`
- `fix/attention-mask-bug`
- `docs/update-readme`
- `issue-123-fix-training-error`

## Architecture Principles

When contributing to OpenAHI, please follow these architectural principles:

1. **No External AI APIs**: Never use external AI providers (Mistral, OpenAI, etc.) as the intelligence engine. The model must be OpenAHI's own.

2. **Modular Design**: Keep components modular and loosely coupled.

3. **Language Boundaries**: Respect the language boundaries:
   - Python: Research, training, experimentation
   - Rust: Runtime, CLI, systems-level functionality
   - C++: Performance-critical inference
   - TypeScript: Web applications, SDKs

4. **Reproducibility**: Ensure that experiments and training are reproducible.

5. **Testing**: Write tests for new functionality.

6. **Documentation**: Document your code and update existing documentation.

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: How to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, hardware, etc.
6. **Logs**: Any relevant error messages or logs

## Security

If you discover a security vulnerability, please see [SECURITY.md](SECURITY.md) for reporting guidelines.

## License

By contributing to OpenAHI, you agree that your contributions will be licensed under the same license as the project (Apache 2.0).

## Recognition

All contributors will be recognized in the project's contributors list. Significant contributions may be highlighted in release notes.

---

Thank you for contributing to OpenAHI!
