# Contributing to vLLM-Omni

Thank you for your interest in contributing to vLLM-Omni! We welcome contributions from the community.

## Quick Start

For detailed contributing guidelines, please visit our [Contributing Documentation](https://vllm-omni.readthedocs.io/en/latest/contributing/).

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/vllm-omni.git
   cd vllm-omni
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Set up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Code Quality

We maintain high code quality standards:

- **Linting**: Code is checked with `ruff`
- **Type Checking**: We use `mypy` in strict mode
- **Testing**: All changes should include tests
- **Pre-commit Hooks**: Run automatically before each commit

Run checks manually:
```bash
# Run linting
ruff check .

# Run type checking
mypy vllm_omni

# Run tests
pytest tests/
```

## Pull Request Process

1. Create a new branch for your changes
2. Make your changes following our coding standards
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request with a clear description

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings to public APIs
- Keep functions focused and modular
- Write descriptive variable names

## Testing

- Write unit tests for new features
- Ensure existing tests pass
- Use pytest markers appropriately:
  - `@pytest.mark.core_model` for core model tests
  - `@pytest.mark.diffusion` for diffusion tests
  - `@pytest.mark.gpu` for GPU-specific tests

## Security

- Never commit secrets or API keys
- Be cautious with pickle usage (see pre-commit checks)
- Report security vulnerabilities privately (see [SECURITY.md](SECURITY.md))

## Community

- Join us on [Slack](https://slack.vllm.ai) in the `#sig-omni` channel
- Participate in discussions on [discuss.vllm.ai](https://discuss.vllm.ai)
- Ask questions and help others

## Questions?

If you have questions about contributing, feel free to:
- Open a discussion on GitHub
- Ask in the Slack channel
- Consult the [full documentation](https://vllm-omni.readthedocs.io/en/latest/contributing/)
