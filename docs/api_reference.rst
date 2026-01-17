API Reference
=============

This section provides the complete API reference for the ``aba_optimiser`` package, automatically generated from the source code docstrings.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   aba_optimiser

Configuration
-------------

.. automodule:: aba_optimiser.config
   :members:
   :show-inheritance:

Optimisers
----------

.. automodule:: aba_optimiser.optimisers.adam
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.optimisers.amsgrad
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.optimisers.lbfgs
   :members:
   :show-inheritance:

Training
--------

.. automodule:: aba_optimiser.training.controller
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.base_controller
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.controller_config
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.controller_helpers
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.configuration_manager
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.data_manager
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.optimisation_loop
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.result_manager
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.scheduler
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.worker_lifecycle
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.worker_manager
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.training.utils
   :members:
   :show-inheritance:

Workers
-------

.. automodule:: aba_optimiser.workers.abstract_worker
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.workers.common
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.workers.optics
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.workers.tracking
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.workers.tracking_position_only
   :members:
   :show-inheritance:

MAD-NG Interface
----------------

.. automodule:: aba_optimiser.mad.base_mad_interface
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.mad.optimising_mad_interface
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.mad.tracking_interface
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.mad.scripts
   :members:
   :show-inheritance:

Physics
-------

.. automodule:: aba_optimiser.physics.bpm_phases
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.physics.deltap
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.physics.dpp_calculation
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.physics.lhc_bends
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.physics.phase_space
   :members:
   :show-inheritance:

Dispersion
----------

.. automodule:: aba_optimiser.dispersion.dispersion_estimation
   :members:
   :show-inheritance:

Momentum Reconstruction
-----------------------

.. automodule:: aba_optimiser.momentum_recon.core
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.transverse
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.dispersive
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.dispersive_measurement
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.momenta
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.momentum_recon.neighbors
   :members:
   :show-inheritance:

Matching
--------

.. automodule:: aba_optimiser.matching.matcher
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.matching.matcher_config
   :members:
   :show-inheritance:

Measurements
------------

.. automodule:: aba_optimiser.measurements.create_datafile
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.create_datafile_b2
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.create_datafile_loop
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.knob_extraction
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.optimise_closed_orbit
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.optimise_squeeze_quads
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.twiss_from_measurement
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.measurements.utils
   :members:
   :show-inheritance:

Filtering
---------

.. automodule:: aba_optimiser.filtering.svd
   :members:
   :show-inheritance:

Model Creator
-------------

.. automodule:: aba_optimiser.model_creator.config
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.model_creator.create_models
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.model_creator.madng_utils
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.model_creator.madx_utils
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.model_creator.tfs_utils
   :members:
   :show-inheritance:

Simulation
----------

.. automodule:: aba_optimiser.simulation.coordinates
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.data_processing
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.mad_setup
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.magnet_perturbations
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.optics
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.simulation.tracking
   :members:
   :show-inheritance:

I/O
---

.. automodule:: aba_optimiser.io.utils
   :members:
   :show-inheritance:

Dataframes
----------

.. automodule:: aba_optimiser.dataframes.utils
   :members:
   :show-inheritance:

Plotting
--------

.. automodule:: aba_optimiser.plotting.strengths
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.plotting.utils
   :members:
   :show-inheritance:

Training Optics
---------------

.. automodule:: aba_optimiser.training_optics.controller
   :members:
   :show-inheritance:

Xsuite
------

.. automodule:: aba_optimiser.xsuite.acd
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.xsuite.action_angle
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.xsuite.env
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.xsuite.monitors
   :members:
   :show-inheritance:

.. automodule:: aba_optimiser.xsuite.tracking
   :members:
   :show-inheritance:
