# sgd-magnet-tuner

This repository demonstrates the application of the Adam optimisation algorithm, based on stochastic gradient descent (SGD), in combination with the MAD-NG simulation code for the identification of magnetic field errors in the Large Hadron Collider (LHC) using beam orbit response. 

## Overview

Magnetic field imperfections in accelerator lattices contribute to orbit distortions and can reduce beam lifetime by enhancing particle loss mechanisms and limiting dynamic aperture. Effective compensation of these errors requires a model that sufficiently reflects the underlying field deviations to guide correction strategies.

### Key Features
- **Gradient-Based Optimisation**: Utilises MAD-NG to provide fast evaluation of the derivatives of particle coordinates with respect to magnet strengths, enabling efficient and effective optimisation.
- **Adam Optimisation Algorithm**: Combines stochastic gradient descent with adaptive learning rates for robust and efficient parameter tuning.
- **Application to LHC Optics**: Demonstrates the method on the Large Hadron Collider optics, addressing real-world challenges in magnetic error modelling.
- **Broad Applicability**: Suitable for a wide class of accelerators requiring improved magnetic error modelling for orbit correction and machine tuning.

## Use Case

This method is designed for:
- Identifying magnetic field errors in accelerators.
- Developing correction strategies to mitigate orbit distortions.
- Enhancing machine tuning and improving beam lifetime.

## Dependencies

- **MAD-NG**: A simulation code for accelerator physics that provides fast derivative evaluations.

## Installation

To use this repository, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/jgray-19/sgd-magnet-tuner.git
   ```
2. Install the required dependencies (specific dependencies for MAD-NG and other tools will need to be added here).

## Usage

1. Configure the simulation environment and input files for your accelerator model.
2. Run the optimisation script to identify magnetic field errors:
   ```bash
   python scripts/run_optimiser.py
   ```
3. Review the output and apply the correction strategies suggested by the model.