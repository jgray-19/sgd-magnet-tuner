Getting started
===============

Installation
------------

Install the project and the documentation extras into a virtual environment::

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[docs]

With dependencies available you can build the HTML documentation::

   cd docs
   make html

The rendered site will be available under ``docs/_build/html/index.html``.

Project overview
----------------

The package is structured around a few core areas:

* :mod:`aba_optimiser.config` stores simulation defaults and file locations.
* :mod:`aba_optimiser.optimisers` implements the optimisation algorithms.
* :mod:`aba_optimiser.simulation` wires together accelerator simulations.
* :mod:`aba_optimiser.training` contains the model training pipelines.

Each module ships with docstrings that are automatically rendered in the API
reference section.
