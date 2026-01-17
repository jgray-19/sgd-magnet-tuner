Getting Started
===============

Installation
------------

Install the project and its dependencies into a virtual environment::

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

For development with testing and documentation tools::

   pip install -e .[test,docs,tracking]

Building Documentation
----------------------

With the documentation extras installed, you can build the HTML documentation::

   cd docs
   make html

The rendered site will be available under ``docs/_build/html/index.html``.

Quick Start
-----------

Running the optimiser::

   python scripts/run_optimiser.py

Optimising energy parameters::

   python scripts/optimise_energy.py

Plotting results::

   python scripts/plot_results.py

Running Tests
-------------

Run the test suite with pytest::

   pytest tests/

Generate a coverage report::

   pytest tests/ --cov=aba_optimiser

Project Overview
----------------

The ``aba_optimiser`` package is structured around several core areas:

* :mod:`aba_optimiser.config` - Configuration constants and simulation defaults
* :mod:`aba_optimiser.optimisers` - Optimisation algorithms (Adam, AMSGrad, L-BFGS)
* :mod:`aba_optimiser.training` - Model training pipelines and orchestration
* :mod:`aba_optimiser.workers` - Parallel worker processes for distributed computation
* :mod:`aba_optimiser.mad` - MAD-NG interface modules for accelerator simulation
* :mod:`aba_optimiser.physics` - Analytical helpers for beam dynamics

Each module ships with docstrings that are automatically rendered in the API
reference section.
