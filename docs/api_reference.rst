API Reference
=============

This section provides the complete API reference for the ``aba_optimiser`` package, automatically generated from the source code docstrings.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   aba_optimiser

Configuration
-------------

.. automodule:: aba_optimiser.config
   :members:
   :undoc-members:
   :show-inheritance:

Optimisers
----------

.. automodule:: aba_optimiser.optimisers.adam
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.optimisers.amsgrad
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.optimisers.lbfgs
   :members:
   :undoc-members:
   :show-inheritance:

Training
--------

.. automodule:: aba_optimiser.training.controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.base_controller
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.controller_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.controller_helpers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.configuration_manager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.data_manager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.optimisation_loop
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.result_manager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.scheduler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.worker_lifecycle
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.worker_manager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.utils
   :members:
   :undoc-members:
   :show-inheritance:

Workers
-------

.. automodule:: aba_optimiser.workers.abstract_worker
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.workers.common
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.workers.optics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.workers.tracking
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.workers.tracking_position_only
   :members:
   :undoc-members:
   :show-inheritance:

MAD-NG Interface
----------------

.. automodule:: aba_optimiser.mad.base_mad_interface
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.mad.optimising_mad_interface
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.mad.tracking_interface
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.mad.scripts
   :members:
   :undoc-members:
   :show-inheritance:

Physics
-------

.. automodule:: aba_optimiser.physics.bpm_phases
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.physics.deltap
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.physics.dpp_calculation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.physics.lhc_bends
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.physics.phase_space
   :members:
   :undoc-members:
   :show-inheritance:

Dispersion
----------

.. automodule:: aba_optimiser.dispersion.dispersion_estimation
   :members:
   :undoc-members:
   :show-inheritance:

Momentum Reconstruction
-----------------------

.. automodule:: aba_optimiser.momentum_recon.core
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.transverse
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.dispersive
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.dispersive_measurement
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.momenta
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.neighbors
   :members:
   :undoc-members:
   :show-inheritance:

Matching
--------

.. automodule:: aba_optimiser.matching.matcher
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.matching.matcher_config
   :members:
   :undoc-members:
   :show-inheritance:

Measurements
------------

.. automodule:: aba_optimiser.measurements.create_datafile
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.create_datafile_b2
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.create_datafile_loop
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.knob_extraction
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.optimise_closed_orbit
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.optimise_squeeze_quads
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.twiss_from_measurement
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.utils
   :members:
   :undoc-members:
   :show-inheritance:

Filtering
---------

.. automodule:: aba_optimiser.filtering.svd
   :members:
   :undoc-members:
   :show-inheritance:

Model Creator
-------------

.. automodule:: aba_optimiser.model_creator.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.model_creator.create_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.model_creator.madng_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.model_creator.madx_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.model_creator.tfs_utils
   :members:
   :undoc-members:
   :show-inheritance:

Simulation
----------

.. automodule:: aba_optimiser.simulation.coordinates
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.data_processing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.mad_setup
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.magnet_perturbations
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.optics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.tracking
   :members:
   :undoc-members:
   :show-inheritance:

I/O
---

.. automodule:: aba_optimiser.io.utils
   :members:
   :undoc-members:
   :show-inheritance:

Dataframes
----------

.. automodule:: aba_optimiser.dataframes.utils
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
--------

.. automodule:: aba_optimiser.plotting.strengths
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.plotting.utils
   :members:
   :undoc-members:
   :show-inheritance:

Training Optics
---------------

.. automodule:: aba_optimiser.training_optics.controller
   :members:
   :undoc-members:
   :show-inheritance:

Xsuite
------

.. automodule:: aba_optimiser.xsuite.acd
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.xsuite.action_angle
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.xsuite.env
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.xsuite.monitors
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aba_optimiser.xsuite.tracking
   :members:
   :undoc-members:
   :show-inheritance:
