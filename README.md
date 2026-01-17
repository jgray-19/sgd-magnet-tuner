# sgd-magnet-tuner

[![Coverage Status](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml/badge.svg)](https://github.com/jgray-19/sgd-magnet-tuner/actions/workflows/coverage.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/jgray-19/sgd-magnet-tuner/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/jgray-19/sgd-magnet-tuner/blob/python-coverage-comment-action-data/htmlcov/index.html)

This repository provides tools for optimising accelerator magnet knob strengths using gradient-based methods with the MAD-NG simulation code.

## Package: `aba_optimiser`

The main package `aba_optimiser` contains the following modules:

### `config`
Configuration constants and dataclasses for the optimisation pipeline. Defines `OptimiserConfig` and `SimulationConfig` for controlling learning rates, convergence criteria, worker distribution, and physical parameters to optimise (energy, quadrupoles, bends).

### `dataframes`
Utilities for working with dataframes in the optimisation workflow.

### `dispersion`
Dispersion estimation at corrector magnets using optics analysis data and MAD-NG tracking. Propagates optics parameters from nearby BPMs using differential algebra tracking.

### `filtering`
Measurement preprocessing and noise suppression routines. Includes ellipse-based outlier rejection, Kalman filtering, and SVD-driven clean-up steps.

### `io`
Input/output utilities for reading and writing data files.

### `mad`
MAD-NG interface modules for accelerator simulation:
- `BaseMadInterface`: Core functionality without automatic setup
- `OptimisationMadInterface`: For accelerator optimisation workflows
- `TrackingMadInterface`: Lightweight interface for tracking simulations

Also contains MAD-NG scripts and LHC sequence definitions.

### `matching`
Tools for matching measured beta functions to a target model by adjusting quadrupole knob strengths. Provides `BetaMatcher` and `MatcherConfig` classes.

### `measurements`
Data acquisition helpers for converting turn-by-turn measurements and optics data into formats expected by the optimisation pipeline. Includes BPM data processing, knob extraction, and closed orbit optimisation.

### `model_creator`
Utilities for creating LHC accelerator models. Handles MAD-NG model initialisation, MAD-X sequence creation, tune matching, and TFS file conversion.

### `momentum_recon`
Momentum reconstruction utilities for transverse and dispersive momentum calculations. Includes:
- Transverse momentum reconstruction from position measurements
- Dispersive momentum calculations using neighboring BPM data
- Noise injection utilities

### `optimisers`
Gradient descent optimiser implementations:
- `adam`: Adam optimiser with adaptive learning rates
- `amsgrad`: AMSGrad variant of Adam
- `lbfgs`: L-BFGS optimiser

### `physics`
Analytical helpers for beam dynamics and accelerator physics. Functions for BPM phase calculations, delta-p calculations, LHC bend parameters, and phase space analysis.

### `plotting`
Visualisation helpers for optimisation diagnostics. Produces plots summarising training metrics, optics measurements, and validation comparisons.

### `simulation`
Components for creating simulated accelerator data. Includes MAD-NG simulation setup, magnet perturbations, optics calculations, tracking coordinate generation, and data processing.

### `training`
Orchestration of the complete optimisation workflow:
- `Controller` and `LHCController`: Main controllers for managing the optimisation process
- `OptimisationLoop`: Coordinated gradient evaluation and training loop
- `WorkerManager` and `WorkerLifecycleManager`: Distributed worker process management
- `DataManager` and `ResultManager`: Data and result handling
- `LRScheduler`: Learning rate scheduling with warmup and decay
- Configuration classes for BPMs, measurements, and sequences

### `training_optics`
Controller specifically for optics function (beta, dispersion) optimisation tasks.

### `workers`
Worker process implementations for parallel computation:
- `TrackingWorker`: Particle tracking (supports 'multi-turn' and 'arc-by-arc' modes)
- `PositionOnlyTrackingWorker`: Position-only tracking without momentum
- `OpticsWorker`: Optics function computations (beta, dispersion)

Workers communicate with the main process via pipes and compute gradients and loss functions.

### `xsuite`
Xsuite integration utilities for AC dipole simulations, particle tracking, environment creation, and monitor data processing.

## Dependencies

Core dependencies:
- **MAD-NG (PyMAD-NG)**: Simulation code for accelerator physics with fast derivative evaluations
- **NumPy**: Numerical computing
- **Pandas & TFS-Pandas**: Data manipulation and TFS file handling
- **Matplotlib**: Plotting and visualisation
- **TensorBoardX**: Training metrics logging
- **Uncertainties**: Error propagation

Optional dependencies:
- **Testing**: pytest, pytest-cov
- **Documentation**: Sphinx, sphinx-autodoc-typehints, myst-parser, furo
- **Tracking**: Xsuite, OMC3, cpymad, pyarrow, psutil
- **CERN**: nxcals

## Installation

Clone the repository and install with pip:

```bash
git clone https://github.com/jgray-19/sgd-magnet-tuner.git
cd sgd-magnet-tuner
pip install -e .
```

For development with all optional dependencies:

```bash
pip install -e .[test,docs,tracking]
```

## Usage

The package provides scripts for various optimisation tasks:

```bash
# Run the main optimiser
python scripts/run_optimiser.py

# Optimise energy parameters
python scripts/optimise_energy.py

# Plot optics and results
python scripts/plot_optics.py
python scripts/plot_results.py
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

With coverage report:

```bash
pytest tests/ --cov=aba_optimiser
```

## Documentation

The project includes Sphinx documentation with automatic API extraction. To build the documentation:

```bash
pip install -e .[docs]
cd docs
make html
```

Open `docs/_build/html/index.html` in a browser to explore usage tutorials and API reference.

## Repository Structure

```
.
├── src/aba_optimiser/    # Main package source code
├── tests/                 # Test suite (physics and training tests)
├── scripts/              # Analysis and plotting scripts
├── examples/             # Example usage scripts
├── docs/                 # Sphinx documentation
├── data/                 # Data files and results
└── mad_scripts/          # MAD-NG scripts
```