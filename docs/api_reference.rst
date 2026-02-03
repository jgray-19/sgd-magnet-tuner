API (short)
===============

Very short guide to the main runtime entry points and what each one does.

1. `Controller` (aba_optimiser.training.controller)
   - Orchestrates optimisation runs (energy, quadrupoles, bends).
   - Inputs: `Accelerator`, `OptimiserConfig`, `SimulationConfig`,
     `SequenceConfig`, `MeasurementConfig`, BPM lists.
   - Call: `estimate, unc = Controller(...).run()`

2. `OpticsController` (aba_optimiser.training_optics.controller)
   - Uses TFS measurement files to match optics via knob adjustments.

3. Configuration dataclasses
   - `OptimiserConfig`: learning rates, epochs, optimiser type.
   - `SimulationConfig`: worker distribution, tracks per worker, flags.
   - `SequenceConfig`, `MeasurementConfig`: define magnet ranges and files.

4. Data utilities
   - `prepare_track_dataframe` (simulation.data_processing): convert
     raw tracking to the expected parquet format.
   - `save_knobs` (io.utils): write matched tune knobs.

Data expected
-------------
- Tracking parquet: columns `turn, name, x, px, y, py, ...`
- Corrector TFS files
- Tune knobs files

Minimal example
---------------

.. code-block:: python

   from aba_optimiser.accelerators import LHC
   from aba_optimiser.config import OptimiserConfig, SimulationConfig
   from aba_optimiser.training.controller import Controller

   accel = LHC(beam=1, sequence_file="lhcb1.seq", optimise_quadrupoles=True)
   opt_cfg = OptimiserConfig(max_epochs=100)
   sim_cfg = SimulationConfig(num_workers=4)
   ctrl = Controller(accel, opt_cfg, sim_cfg, None, None, [], [])
   estimate, unc = ctrl.run()

See tests
--------
The tests in `tests/training/` are compact, runnable examples of each workflow.

Autosummary
-----------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   aba_optimiser
