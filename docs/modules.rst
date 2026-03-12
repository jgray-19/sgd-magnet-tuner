Package Overview
================

``aba_optimiser`` is organised around a small number of runtime layers. The
sections below explain what each package is for and when you would normally
read or extend it.

Core Runtime
------------

``aba_optimiser.training``
   Coordinates end-to-end optimisation runs. This is where controllers,
   batching, worker orchestration, result handling, and learning-rate
   scheduling live.

``aba_optimiser.workers``
   Multiprocessing worker implementations used by the training controllers.
   Tracking workers evaluate loss and gradients from turn-by-turn data, while
   optics workers evaluate phase/beta targets.

``aba_optimiser.mad``
   Interfaces to MAD-NG and the bundled MAD scripts. These modules are
   responsible for loading sequences, defining observable ranges, discovering
   knobs, and executing the low-level accelerator computations used by the
   workers.

``aba_optimiser.accelerators``
   Machine-specific behaviour such as sequence conventions, monitor naming,
   available optimisation targets, and model-normalisation rules.


Optimisation Support
--------------------

``aba_optimiser.config``
   Shared dataclasses and constants that describe optimiser settings,
   simulation settings, project paths, and common defaults.

``aba_optimiser.optimisers``
   Stateful optimiser implementations used by the training loop. The project
   currently provides Adam, AMSGrad, and a line-search-free L-BFGS variant.

``aba_optimiser.noise``
   BPM noise tables and helper functions for assigning or applying variances in
   simulated datasets.

``aba_optimiser.io`` and ``aba_optimiser.dataframes``
   Small utility packages for reading/writing result tables and filtering named
   rows in measurement/model tables.


Measurement and Analysis Scripts
--------------------------------

``aba_optimiser.measurements``
   Script-oriented workflows for converting operational measurements into the
   parquet/TFS inputs expected by the optimisation pipeline, plus campaign
   specific entry points for squeeze studies and closed-orbit analysis.

``aba_optimiser.matching``
   Matching utilities for optics-style least-squares studies outside the main
   worker-driven optimisation loop.

``aba_optimiser.plotting``
   Reusable plotting helpers for optimisation diagnostics and reporting.

``aba_optimiser.simulation`` and ``aba_optimiser.model_creator``
   Tools for generating synthetic tracking data, model files, and related
   accelerator setup artefacts.


Suggested Reading Order
-----------------------

For a first pass through the codebase, start with:

1. ``aba_optimiser.config`` for the main runtime knobs.
2. ``aba_optimiser.training.controller`` for the orchestration entry point.
3. ``aba_optimiser.training.optimisation_loop`` for epoch and batch logic.
4. ``aba_optimiser.workers.tracking`` or ``aba_optimiser.workers.optics`` for
   the actual worker-side computation.
5. ``aba_optimiser.mad.optimising_mad_interface`` for the MAD-NG setup layer.
