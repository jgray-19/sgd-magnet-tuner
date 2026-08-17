API Reference
=============

This reference is limited to modules with direct automated test coverage and to
runtime entry points that are exercised in the current suite. Script-heavy,
campaign-specific, or mostly skipped areas are intentionally not published
here yet.

Included Scope
--------------

The documented surface currently focuses on:

1. Accelerator definitions and shared runtime configuration.
2. MAD interface classes used by the optimisation loop.
3. Controller, worker-management, and worker payload APIs.
4. Optimiser implementations and core numerical helpers with active tests.
5. A small set of measurement and utility modules with direct unit tests.

Omitted For Now
---------------

The following areas are intentionally excluded from the published reference
because coverage is partial, heavily import-gated, or currently centred on
workflow scripts rather than stable reusable API:

* most of ``aba_optimiser.measurements``
* ``aba_optimiser.matching``
* most of ``aba_optimiser.simulation``
* physics modules whose tests are skipped or rely on optional environments


Primary Entry Points
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   aba_optimiser.accelerators.Accelerator
   aba_optimiser.accelerators.LHC
   aba_optimiser.accelerators.PSB
   aba_optimiser.accelerators.SPS
   aba_optimiser.accelerators.instantiate_accelerator_from
   aba_optimiser.config.OptimiserConfig
   aba_optimiser.config.SimulationConfig
   aba_optimiser.training.Controller
   aba_optimiser.training.BaseController
   aba_optimiser.training.MeasurementConfig
   aba_optimiser.training.SequenceConfig
   aba_optimiser.training.OutputConfig


Accelerators And Configuration
------------------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.accelerators
   aba_optimiser.accelerators.base
   aba_optimiser.accelerators.lhc
   aba_optimiser.accelerators.psb
   aba_optimiser.accelerators.sps
   aba_optimiser.config


MAD Interface Layer
-------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.mad
   aba_optimiser.mad.aba_mad_interface
   aba_optimiser.mad.optimising_mad_interface


Optimisation Runtime
--------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.training
   aba_optimiser.training.base_controller
   aba_optimiser.training.config
   aba_optimiser.training.controller
   aba_optimiser.training.config.helpers
   aba_optimiser.training.config.manager
   aba_optimiser.training.config.models
   aba_optimiser.training.config.tracking
   aba_optimiser.training.data_manager
   aba_optimiser.training.optimisation
   aba_optimiser.training.optimisation.loop
   aba_optimiser.training.result_manager
   aba_optimiser.training.optimisation.scheduler
   aba_optimiser.training.workers
   aba_optimiser.training.workers.lifecycle
   aba_optimiser.training.workers.manager
   aba_optimiser.training.workers.payloads
   aba_optimiser.training.workers.setup
   aba_optimiser.training.workers.validation


Workers
-------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.workers
   aba_optimiser.workers.abstract_worker
   aba_optimiser.workers.common
   aba_optimiser.workers.tracking
   aba_optimiser.workers.tracking_position_only
   aba_optimiser.workers.tracking_validation


Optimisers And Numerical Helpers
--------------------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.optimisers.adam
   aba_optimiser.optimisers.amsgrad
   aba_optimiser.optimisers.lbfgs
   aba_optimiser.physics.deltap
   aba_optimiser.dataframes.utils
   aba_optimiser.io.utils


Tested Measurement Utilities
----------------------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.measurements.b2_errors
   aba_optimiser.measurements.create_datafile
   aba_optimiser.measurements.loading
   aba_optimiser.measurements.plot_quad_diffs_and_phases
   aba_optimiser.measurements.sequence
   aba_optimiser.measurements.utils
