Module Reference
================

This section provides detailed information about each module in the ``aba_optimiser`` package.

Config
------

Configuration constants and dataclasses for the optimisation pipeline.

**Key Components:**

* ``OptimiserConfig`` - Controls learning rates, convergence criteria, and optimiser type (Adam, AMSGrad, L-BFGS)
* ``SimulationConfig`` - Manages worker distribution, tracking data, and physical parameters to optimise (energy, quadrupoles, bends)

**Optimisation Parameters:**

* Learning rate scheduling with warmup and decay phases
* Gradient convergence thresholds
* Worker parallelization settings
* Toggle for optimising energy, quadrupoles, and bends independently

**Configuration Constants:**

* ``DPP_OPTIMISER_CONFIG`` - Optimiser settings for momentum deviation (dp/p) optimisation
* ``DPP_SIMULATION_CONFIG`` - Simulation settings for momentum optimisation
* ``QUAD_OPTIMISER_CONFIG`` - Optimiser settings for quadrupole strength optimisation
* ``QUAD_SIMULATION_CONFIG`` - Simulation settings for quadrupole optimisation


Dataframes
----------

Utilities for working with dataframes in the optimisation workflow. Provides helper functions for manipulating and transforming measurement data stored in pandas DataFrames.


Dispersion
----------

Dispersion estimation at corrector magnets using optics analysis data and MAD-NG tracking.

**Features:**

* Estimates horizontal and vertical dispersion at corrector locations
* Propagates optics parameters from nearby BPMs
* Uses MAD-NG's differential algebra tracking capabilities for accurate propagation
* Supports multi-processing for efficient computation across many correctors


Filtering
---------

Measurement preprocessing and noise suppression routines.

**Techniques:**

* Ellipse-based outlier rejection for removing bad measurements
* Kalman filtering for temporal smoothing
* SVD-driven clean-up steps to identify and remove systematic errors
* Statistical filtering to improve signal-to-noise ratio


IO
--

Input/output utilities for reading and writing data files.

**Functionality:**

* Reading and writing TFS (Table File System) files
* Parquet file handling for large datasets
* Configuration file parsing
* Result serialization and deserialization


MAD
---

MAD-NG interface modules for accelerator simulation.

**Interface Classes:**

* ``BaseMadInterface`` - Core functionality without automatic setup, provides low-level access to MAD-NG
* ``OptimisationMadInterface`` - Specialized for accelerator optimisation workflows with gradient computation
* ``TrackingMadInterface`` - Lightweight interface for tracking simulations

**Additional Components:**

* MAD-NG scripts for LHC lattice initialization
* LHC sequence definitions (beam 1 and beam 2)
* Utility functions for MAD-NG object manipulation


Matching
--------

Tools for matching measured beta functions to a target model by adjusting quadrupole knob strengths.

**Classes:**

* ``BetaMatcher`` - Main class for beta function matching using least-squares optimization
* ``MatcherConfig`` - Configuration dataclass for matcher parameters

**Capabilities:**

* Matches beta functions at specified locations
* Adjusts quadrupole knob strengths to minimize differences
* Supports constraints on knob strength ranges
* Provides detailed diagnostics of matching quality


Measurements
------------

Data acquisition helpers for converting turn-by-turn measurements and optics data into formats expected by the optimisation pipeline.

**Functionality:**

* BPM data processing and cleaning
* Knob extraction from measurement files
* Closed orbit optimisation
* Turn-by-turn data analysis
* Creation of datafiles for training from raw measurements
* Quadrupole strength optimisation for squeeze scenarios


Model Creator
-------------

Utilities for creating LHC accelerator models.

**Features:**

* MAD-NG model initialisation with proper optics and errors
* MAD-X sequence creation and manipulation
* Tune matching to desired working point
* TFS file conversion between formats
* Support for different LHC optics years and configurations

**Key Functions:**

* ``create_lhc_model`` - Creates complete LHC model with specified parameters
* ``initialise_madng_model`` - Initializes MAD-NG with LHC sequence
* ``match_model_tunes`` - Matches model to target tunes using knobs
* ``convert_tfs_to_madx`` - Converts TFS files to MAD-X format


Momentum Recon
--------------

Momentum reconstruction utilities for transverse and dispersive momentum calculations.

**Reconstruction Methods:**

* **Transverse momentum reconstruction** - Calculates momenta from position measurements using local optics
* **Dispersive momentum reconstruction** - Uses neighboring BPM data to estimate momentum deviation
* **Combined approach** - Merges both methods for improved accuracy

**Utilities:**

* Noise injection for testing and validation
* Momentum calculation from measurement data
* Error propagation through momentum reconstruction


Optimisers
----------

Gradient descent optimiser implementations.

**Available Optimisers:**

* **Adam** - Adaptive learning rate optimiser with momentum and RMSprop
* **AMSGrad** - Variant of Adam with improved convergence guarantees
* **L-BFGS** - Limited-memory Broyden-Fletcher-Goldfarb-Shanno quasi-Newton optimiser

**Features:**

* Learning rate scheduling support
* Gradient clipping and smoothing
* Convergence detection
* State persistence for checkpointing


Physics
-------

Analytical helpers for beam dynamics and accelerator physics.

**Functions:**

* BPM phase calculations between measurement points
* Delta-p (momentum deviation) calculations
* LHC bend parameters and field errors
* Phase space analysis and transformations
* Momentum calculations from positions using optics functions


Plotting
--------

Visualisation helpers for optimisation diagnostics.

**Plot Types:**

* Training metrics over epochs (loss, gradients)
* Optics measurements (beta functions, dispersion)
* Validation comparisons between measured and simulated data
* Magnet strength evolution during optimisation
* Phase space plots
* Convergence diagnostics


Simulation
----------

Components for creating simulated accelerator data.

**Capabilities:**

* MAD-NG simulation setup and configuration
* Magnet perturbations (quadrupoles, dipoles, correctors)
* Beam optics calculations
* Tracking coordinate generation
* Data processing and formatting for training

**Workflow:**

1. Set up MAD-NG environment with LHC lattice
2. Apply magnet perturbations (errors)
3. Track particles through the lattice
4. Calculate optics functions
5. Process and format data for optimisation


Training
--------

Orchestration of the complete optimisation workflow.

**Core Classes:**

* ``Controller`` - Main controller for managing the optimisation process
* ``LHCController`` - LHC-specific controller with additional features
* ``OptimisationLoop`` - Coordinated gradient evaluation and training loop
* ``WorkerManager`` - Manages distributed worker processes
* ``WorkerLifecycleManager`` - Handles worker startup and shutdown
* ``DataManager`` - Handles data loading, preprocessing, and batching
* ``ResultManager`` - Manages result collection and persistence
* ``LRScheduler`` - Learning rate scheduling with warmup and decay

**Configuration:**

* ``BPMConfig`` - Configuration for BPM selection and ranges
* ``MeasurementConfig`` - Configuration for measurement data processing
* ``SequenceConfig`` - Configuration for accelerator sequence parameters

**Workflow:**

1. Initialize controller with configuration
2. Set up worker processes for parallel computation
3. Load and preprocess measurement data
4. Run optimisation loop with gradient updates
5. Save results and diagnostics


Training Optics
---------------

Controller specifically for optics function (beta, dispersion) optimisation tasks.

**Features:**

* Optimises beta functions at specified locations
* Optimises dispersion functions
* Uses optics measurements as constraints
* Supports both local and global optics corrections


Workers
-------

Worker process implementations for parallel computation.

**Worker Types:**

* ``TrackingWorker`` - Particle tracking with full phase space (positions and momenta)
  
  * Supports 'multi-turn' mode for tracking over multiple turns
  * Supports 'arc-by-arc' mode for segmented tracking
  
* ``PositionOnlyTrackingWorker`` - Position-only tracking without momentum (faster for some applications)
* ``OpticsWorker`` - Optics function computations (beta, dispersion) without tracking

**Data Structures:**

* ``TrackingData`` - Input data for tracking workers
* ``OpticsData`` - Input data for optics workers
* ``WorkerConfig`` - Configuration for all worker types
* ``WeightProcessor`` - Handles weighting of measurements in loss function

**Communication:**

* Workers communicate with main process via pipes
* Compute gradients and loss functions in parallel
* Support for batched computation to improve efficiency


Xsuite
------

Xsuite integration utilities for particle tracking and simulations.

**Features:**

* AC dipole (ACD) simulation and tracking
* Particle tracking without AC dipole
* Xsuite environment creation and initialization
* Monitor insertion at pattern-matched locations
* Tracking data processing and DataFrame conversion
* Twiss parameter conversion between Xsuite and MAD-NG formats

**Key Functions:**

* ``insert_ac_dipole`` - Adds AC dipole elements to lattice
* ``run_acd_track`` - Runs tracking with AC dipole excitation
* ``create_xsuite_environment`` - Sets up complete Xsuite tracking environment
* ``line_to_dataframes`` - Converts Xsuite line monitors to pandas DataFrames
* ``xsuite_tws_to_ng`` - Converts Xsuite Twiss to MAD-NG format
