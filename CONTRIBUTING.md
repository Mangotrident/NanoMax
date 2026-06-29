# Contributing to AND-CP

First off, thank you for taking the time to contribute! We welcome contributions from computational neuroscientists, machine learning engineers, and software developers alike.

## Code of Conduct

This project and everyone participating in it is governed by the [AND-CP Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs
* Check the current Issues list to ensure the bug has not already been reported.
* Open a new issue with a clear title, description, steps to reproduce, and any relevant traceback or logging output.

### Requesting Features
* Open an issue describing the feature, the use-case, and potential design considerations.

### Submitting Pull Requests
1. Fork the repository and create your branch from `main`.
2. Write clean code conforming to the PEP 8 standard.
3. Add type hints and docstrings for all public classes, functions, and methods.
4. Ensure that the test suite passes:
   ```bash
   PYTHONPATH=. python -m unittest tests/test_pipeline.py
   ```
5. Open the PR with a comprehensive description of the changes.

## Development Style Guide

* **Linting**: We enforce code formatting via `black` and linting via `ruff`.
* **Testing**: All new features must include unit or integration tests in the `tests/` directory.
* **Architecture**: Maintain a modular, decoupled interface separating loaders, preprocessors, models, simulators, and live servers.
