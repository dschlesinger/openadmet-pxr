# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[test]"

# Format
black src/ tests/

# Lint
flake8 src/ tests/
pylint src/

# Type check
pyright

# Run unit tests (default — excludes integration/spark)
pytest

# Run a single test
pytest tests/test_methods.py::test_hello -v

# Run integration tests
pytest -m integration

# Run all tests via tox (unit / integration / spark / all)
tox
tox -e integration
tox -e spark
tox -e all
```

## Architecture

This is the **OpenADMET PXR challenge** dataset package. Its primary job is to expose training/test splits of the Pregnane X Receptor (PXR) dataset from HuggingFace, and provide a thin Python API around them.

**Source layout** (`src/` on `PYTHONPATH` via pytest):
- `src/python_package/` — installable package (`python_package`). Version lives in `__init__.py`.
- `src/scripts/download.py` — loads the PXR challenge CSV splits from HuggingFace Hub into Polars DataFrames. This is the core functional code; `hello_world.py` is scaffolding from the template.

**Dependencies**:
- Runtime deps go in `requirements.txt` (currently: `polars`); `pyproject.toml` reads them dynamically via `tool.setuptools.dynamic`.
- Optional deps (pyspark, test tools) remain in `[project.optional-dependencies]` in `pyproject.toml`.

**Testing constraints**:
- 100% branch coverage is enforced — `pytest-cov` will fail below that threshold.
- Test markers: `unit`, `integration`, `spark`, `gpu`, `slow`, `notebooks`. The default pytest run excludes `integration` and `spark`.
- Tests auto-detect by name convention: files with `_int_` in the name are auto-marked `integration`; `spark` tests require the `pyspark` extra.

**Code style**:
- Max line length: 120 characters (Black, Flake8, Pylint all agree).
- Pylint max function args: 5; max attributes per class: 7.
- Pyright type checking is enabled — add type annotations to all public APIs.
