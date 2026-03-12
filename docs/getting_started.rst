Getting Started
===============

This page gives a practical overview of how to install ``aba_optimiser``,
prepare inputs, and identify the main entry points in the codebase.

Installation
------------

Create a virtual environment and install the package:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

For development with tests, tracking extras, and documentation tools:

.. code-block:: bash

   pip install -e .[test,docs,tracking]


Main Concepts
-------------

``Accelerator``
   Encapsulates machine-specific behaviour such as sequence files, knob
   families, and BPM naming conventions.

``OptimiserConfig``
   Controls epochs, optimiser type, learning rates, and convergence criteria.

``SimulationConfig``
   Controls worker counts, batching, tracking mode, and runtime physics flags.

``MeasurementConfig`` / ``SequenceConfig``
   Describe the input files and the magnet/BPM ranges used by a controller.

``Controller``
   Coordinates the full optimisation workflow: data loading, worker startup,
   epoch/batch execution, and result collection.


Typical Workflow
----------------

1. Prepare a measurement parquet file and any supporting corrector/tune-knob
   files.
2. Instantiate an accelerator, usually :class:`aba_optimiser.accelerators.LHC`.
3. Build optimiser, simulation, measurement, and sequence configuration
   dataclasses.
4. Construct :class:`aba_optimiser.training.controller.Controller`.
5. Call ``run()`` to obtain estimated knob strengths and uncertainties.


Minimal Example
---------------

.. code-block:: python

   from pathlib import Path

   from aba_optimiser.accelerators import LHC
   from aba_optimiser.config import OptimiserConfig, SimulationConfig
   from aba_optimiser.training.controller import Controller
   from aba_optimiser.training.controller_config import MeasurementConfig, SequenceConfig

   accelerator = LHC(
       beam=1,
       beam_energy=6800,
       sequence_file=Path("lhcb1.seq"),
       optimise_quadrupoles=True,
   )

   optimiser_config = OptimiserConfig(max_epochs=100, optimiser_type="lbfgs")
   simulation_config = SimulationConfig(
       tracks_per_worker=10,
       num_workers=4,
       num_batches=1,
       use_fixed_bpm=True,
   )
   sequence_config = SequenceConfig(magnet_range="BPM.9R1.B1/BPM.9L2.B1")
   measurement_config = MeasurementConfig(
       measurement_files=Path("tracking_data.parquet"),
       corrector_files=Path("correctors.tfs"),
       tune_knobs_files=Path("tune_knobs.txt"),
       flattop_turns=1000,
       bunches_per_file=1,
   )

   controller = Controller(
       accelerator,
       optimiser_config,
       simulation_config,
       sequence_config,
       measurement_config,
       bpm_start_points=["BPM.9R1.B1"],
       bpm_end_points=["BPM.9L2.B1"],
       show_plots=False,
   )
   estimate, uncertainties = controller.run()


Tests and Examples
------------------

The most useful runnable examples are in ``tests/training/``. They exercise the
same controller and worker code paths used in production-style runs.

Useful commands:

.. code-block:: bash

   pytest tests/training/
   pytest tests/ -k quadrupole


Building the Docs
-----------------

With the docs extras installed, build the HTML site with:

.. code-block:: bash

   cd docs
   make html

The rendered output will be written to ``docs/_build/html/index.html``.


Where To Read Next
------------------

* :doc:`modules` for a package-by-package overview.
* :doc:`api_reference` for the public runtime entry points.
