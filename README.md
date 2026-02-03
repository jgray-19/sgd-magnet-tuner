# sgd-magnet-tuner

[![Coverage Status](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml/badge.svg)](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml)
![Coverage](https://raw.githubusercontent.com/jgray-19/sgd-magnet-tuner/python-coverage-comment-action-data/badge.svg)

Tools for optimising accelerator magnet knob strengths using gradient-based
methods with MAD-NG. This README is short and focused — see the docs for
full details.

## Package overview

High-level modules (concise):

- `config` — configuration dataclasses and defaults
- `training` — `Controller` runtime and orchestration (energy, quads, bends)
- `training_optics` — optics-specific matching controller
- `simulation` / `mad` — model creation and tracking utilities
- `measurements` / `dataframes` / `filtering` — measurement IO and cleaning
- `optimisers` — Adam / AMSGrad / L-BFGS implementations
- `io` / `plotting` / `matching` — helpers and utilities

Use the tests in `tests/training/` as compact examples of real workflows.

## Dependencies (external projects)

This project uses helper packages maintained in related repositories; install
them before running the end-to-end workflows:

- xtrack_tools: https://github.com/jgray-19/xtrack_tools
- tmom-recon:  https://github.com/jgray-19/tmom-recon

Install via pip from GitHub, for example::

```bash
pip install git+https://github.com/jgray-19/xtrack_tools.git
pip install git+https://github.com/jgray-19/tmom-recon.git
```

## Installation

Clone and install in editable mode::

```bash
git clone https://github.com/jgray-19/sgd-magnet-tuner.git
cd sgd-magnet-tuner
pip install -e .
```

For development (tests + docs):

```bash
pip install -e .[test,docs,tracking]
```

## Quick usage

Run the main scripts (examples):

```bash
python scripts/run_optimiser.py
python scripts/optimise_energy.py
python scripts/plot_results.py
```

## Tests

Run tests with pytest::

```bash
pytest tests/
pytest tests/ --cov=aba_optimiser
```

## Docs

Build docs::

```bash
pip install -e .[docs]
cd docs && make html
```

View at `docs/_build/html/index.html`.

## Installation

Clone and install in editable mode::

```bash
git clone https://github.com/jgray-19/sgd-magnet-tuner.git
cd sgd-magnet-tuner
pip install -e .
```

For development (tests + docs):

```bash
pip install -e .[test,docs,tracking]
```

## Quick usage

Run the main scripts (examples):

```bash
python scripts/run_optimiser.py
python scripts/optimise_energy.py
python scripts/plot_results.py
```

## Tests

Run tests with pytest::

```bash
pytest tests/
pytest tests/ --cov=aba_optimiser
```

## Docs

Build docs::

```bash
pip install -e .[docs]
cd docs && make html
```

View at `docs/_build/html/index.html`.