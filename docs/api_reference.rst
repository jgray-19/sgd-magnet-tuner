API Reference
=============

This section focuses on the public runtime entry points that most users need to
understand before running or extending the project.

Main Entry Points
-----------------

``aba_optimiser.training.controller.Controller``
   Main controller for worker-based tracking optimisation.

``aba_optimiser.training_optics.controller.OpticsController``
   Optics-focused controller for beta/phase style matching problems.

``aba_optimiser.config.OptimiserConfig`` and ``aba_optimiser.config.SimulationConfig``
   Dataclasses that define optimisation behaviour, batching, and worker setup.

``aba_optimiser.accelerators.LHC`` / ``aba_optimiser.accelerators.SPS``
   Machine-specific accelerator definitions.


Typical Workflow
----------------

1. Choose an accelerator and a magnet range.
2. Build optimiser and simulation configuration dataclasses.
3. Prepare measurement files or generated parquet tracking data.
4. Construct a controller and call ``run()``.

The tests under ``tests/training/`` are compact runnable examples of that
workflow.


Public Modules
--------------

.. autosummary::
   :toctree: _autosummary

   aba_optimiser.config
   aba_optimiser.accelerators.lhc
   aba_optimiser.accelerators.sps
   aba_optimiser.mad.optimising_mad_interface
   aba_optimiser.optimisers.adam
   aba_optimiser.optimisers.amsgrad
   aba_optimiser.optimisers.lbfgs
   aba_optimiser.training.controller
   aba_optimiser.training.controller_config
   aba_optimiser.training.optimisation_loop
   aba_optimiser.training.scheduler
   aba_optimiser.training.worker_manager
   aba_optimiser.training_optics.controller
   aba_optimiser.workers.common
   aba_optimiser.workers.tracking
   aba_optimiser.workers.tracking_position_only
   aba_optimiser.workers.optics
