aba_optimiser documentation
===========================

``aba_optimiser`` is a worker-based optimisation toolkit for accelerator magnet
studies. The current documentation is intentionally narrow: it focuses on the
API surface that is exercised by the automated tests and used by the main
controller-driven runtime.

Use this site as a reference for:

* accelerator definitions and runtime configuration dataclasses
* MAD interface classes used to construct optimisation problems
* controller, worker, and optimiser APIs
* tested utility modules that support data preparation and analysis

Workflow guides and campaign-specific scripts are intentionally left out of the
published docs until they have stronger validation coverage.

GitHub Dependencies
-------------------

Some parts of the repository rely on companion packages installed directly from
GitHub rather than PyPI-only dependencies:

* ``pymadng-utils`` provides shared accelerator abstractions plus MAD/MAD-X
  helper utilities such as knob file IO and interface glue used throughout the
  core runtime.
* ``tmom-recon`` provides transverse and longitudinal momentum reconstruction,
  AC-dipole measurement helpers, and optics reconstruction routines used by the
  measurement and optics-oriented code paths.
* ``xtrack-tools`` provides tracking helpers, environment initialisation, and
  dataframe conversion utilities used by the higher-fidelity controller and
  simulation tests.

These dependencies are important because the tested end-to-end workflows in
this repository are built around a larger accelerator-tooling stack rather than
standalone numerical routines.

Companion documentation:

* ``sgd-magnet-tuner``: `jgray-19.github.io/sgd-magnet-tuner <https://jgray-19.github.io/sgd-magnet-tuner/>`_
* ``pymadng-utils``: `jgray-19.github.io/pymadng-utils <https://jgray-19.github.io/pymadng-utils/>`_
* ``tmom-recon``: `jgray-19.github.io/tmom-recon <https://jgray-19.github.io/tmom-recon/>`_
* ``xtrack_tools``: `jgray-19.github.io/xtrack_tools <https://jgray-19.github.io/xtrack_tools/>`_

.. toctree::
   :maxdepth: 2
   :caption: API

   api_reference
