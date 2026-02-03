Getting Started
===============

Welcome to ``aba_optimiser``! This package provides tools for stochastic optimization of accelerator magnet knobs using machine learning techniques. Whether you're optimizing beam energy, quadrupole strengths, bend fields, or matching optics, this guide will help you get up and running quickly.

Prerequisites
-------------

Before installing ``aba_optimiser``, ensure you have:

* **Python 3.9+**: The package requires Python 3.9 or higher
* **MAD-NG**: Accelerator simulation software (MAD-NG) for model creation
* **Git**: For cloning repositories and version control
* **Virtual environment tools**: ``venv`` (built-in) or ``conda``

Installation
------------

Install the project and its dependencies into a virtual environment::

   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install the package
   pip install -e .

For development with testing and documentation tools::

   pip install -e .[test,docs,tracking]


Quick Start Examples
--------------------

Here are some quick examples to get you started. For detailed optimization workflows, see the :doc:`optimization guide <api_reference>`.

**Basic Energy Optimization**

.. code-block:: python

   from aba_optimiser.accelerators import LHC
   from aba_optimiser.config import OptimiserConfig, SimulationConfig
   from aba_optimiser.training.controller import Controller
   from aba_optimiser.training.controller_config import MeasurementConfig, SequenceConfig

   # Set up accelerator
   accelerator = LHC(beam=1, sequence_file="lhcb1.seq", optimise_energy=True)

   # Configure optimization
   optimiser_config = OptimiserConfig(max_epochs=100, max_lr=1e-6)
   simulation_config = SimulationConfig(num_workers=4, optimise_momenta=True)

   # Define measurements
   measurement_config = MeasurementConfig(
       measurement_files="tracking_data.parquet",
       corrector_files="correctors.tfs",
       tune_knobs_files="tune_knobs.txt",
       flattop_turns=1000
   Getting Started (Short)
   =======================

   Quick, minimal guide to start using ``aba_optimiser``.

   Install
   -------

   Create a virtual environment and install the package:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate
      pip install -e .[test,docs,tracking]

   Verify installation::

      python -c "import aba_optimiser; print('OK')"

   Run a simple optimization
   ------------------------

   Minimal example (energy or quadrupole optimization uses the same Controller API):

   .. code-block:: python

      from aba_optimiser.accelerators import LHC
      from aba_optimiser.config import OptimiserConfig, SimulationConfig
      from aba_optimiser.training.controller import Controller
      from aba_optimiser.training.controller_config import MeasurementConfig, SequenceConfig

      accel = LHC(beam=1, sequence_file="lhcb1.seq", optimise_energy=True)
      opt_cfg = OptimiserConfig(max_epochs=100, max_lr=1e-6)
      sim_cfg = SimulationConfig(num_workers=4, optimise_momenta=True)

      meas_cfg = MeasurementConfig(measurement_files="data.parquet",
                                   corrector_files="correctors.tfs",
                                   tune_knobs_files="tune_knobs.txt",
                                   flattop_turns=1000)

      ctrl = Controller(accel, opt_cfg, sim_cfg, SequenceConfig(), meas_cfg, [], [])
      estimate, uncertainty = ctrl.run()

   Commands
   --------

   Run tests::

   .. code-block:: bash

      pytest tests/

   Build docs::

   .. code-block:: bash

      cd docs && make html

   Where to go next
   -----------------

   - See the short ``Optimization Guide`` (:doc:`api_reference`) for how to choose configs and prepare data.
   - Use the tests in ``tests/training/`` as compact examples of real workflows.

   That's it — minimal steps to install and run a basic optimization.
   # Run specific test categories
   pytest tests/training/  # Training-related tests
   pytest tests/mad/       # MAD interface tests
   pytest tests/ -k "energy"  # Tests containing "energy"

Building Documentation
----------------------

With the documentation extras installed, you can build the HTML documentation::

   cd docs
   make html

The rendered site will be available under ``docs/_build/html/index.html``.

You can also build other formats::

   make pdf    # PDF documentation
   make epub   # EPUB format

Project Structure
-----------------

The ``aba_optimiser`` package is organized into several key modules:

**Core Optimization**

* :mod:`aba_optimiser.config` - Configuration dataclasses and defaults
* :mod:`aba_optimiser.optimisers` - Optimization algorithms (Adam, AMSGrad, L-BFGS)
* :mod:`aba_optimiser.training` - Training pipelines and controller classes
* :mod:`aba_optimiser.training_optics` - Optics-specific optimization

**Simulation & Physics**

* :mod:`aba_optimiser.mad` - MAD-NG interface for accelerator simulation
* :mod:`aba_optimiser.physics` - Beam dynamics calculations
* :mod:`aba_optimiser.simulation` - Tracking and optics simulation
* :mod:`aba_optimiser.xsuite` - Xsuite integration utilities

**Data Handling**

* :mod:`aba_optimiser.measurements` - Data acquisition and processing
* :mod:`aba_optimiser.io` - File I/O utilities
* :mod:`aba_optimiser.dataframes` - Data manipulation helpers
* :mod:`aba_optimiser.filtering` - Signal processing and noise reduction

**Utilities**

* :mod:`aba_optimiser.plotting` - Visualization tools
* :mod:`aba_optimiser.matching` - Optics matching algorithms
* :mod:`aba_optimiser.model_creator` - Model generation utilities
* :mod:`aba_optimiser.momentum_recon` - Momentum reconstruction
* :mod:`aba_optimiser.dispersion` - Dispersion estimation

Next Steps
----------

Now that you're set up, here are some recommended next steps:

1. **Run the Tests**: Familiarize yourself with the testing framework
2. **Check the API Reference**: Complete documentation for all modules

For questions or issues, please check the project repository or contact the maintainers.
