# sgd-magnet-tuner

[![Coverage Status](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml/badge.svg)](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/jgray-19/sgd-magnet-tuner/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/jgray-19/sgd-magnet-tuner/blob/python-coverage-comment-action-data/htmlcov/index.html)

Tools for optimising accelerator magnet knob strengths using gradient-based methods with MAD-NG.

## Modules

Config
------
Configuration constants and dataclasses for the optimisation pipeline.

Dataframes
----------
Utilities for working with dataframes in the optimisation workflow.

Dispersion
----------
Dispersion estimation at corrector magnets using MAD-NG tracking.

Filtering
---------
Measurement preprocessing and noise suppression routines.

IO
--
Input/output utilities for reading and writing data files.

MAD
---
MAD-NG interface modules for accelerator simulation.

Matching
--------
Tools for matching beta functions to target models.

Measurements
------------
Data acquisition helpers for turn-by-turn measurements and optics data.

Model Creator
-------------
Utilities for creating LHC accelerator models.

Momentum Recon
--------------
Momentum reconstruction utilities for transverse and dispersive calculations.

Optimisers
----------
Gradient descent optimiser implementations (Adam, AMSGrad, L-BFGS).

Physics
-------
Analytical helpers for beam dynamics and accelerator physics.

Plotting
--------
Visualisation helpers for optimisation diagnostics.

Simulation
----------
Components for creating simulated accelerator data.

Training
--------
Orchestration of the complete optimisation workflow.

Training Optics
---------------
Controller for optics function optimisation tasks.

Workers
-------
Worker process implementations for parallel computation.

Xsuite
------
Xsuite integration utilities for particle tracking and simulations.

## Installation

```bash
git clone https://github.com/jgray-19/sgd-magnet-tuner.git
cd sgd-magnet-tuner
pip install -e .
```

For development:
```bash
pip install -e .[test,docs,tracking]
```

## Usage

```bash
python scripts/run_optimiser.py
python scripts/optimise_energy.py
python scripts/plot_results.py
```

## Testing

```bash
pytest tests/
pytest tests/ --cov=aba_optimiser
```

## Documentation

Build the documentation:
```bash
pip install -e .[docs]
cd docs
make html
```

View at `docs/_build/html/index.html`.