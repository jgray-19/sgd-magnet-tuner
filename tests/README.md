# Unit Tests for BPM Phases Module

This directory contains comprehensive unit tests for the BPM (Beam Position Monitor) phases functionality in the sgd-magnet-tuner project.

## Test Coverage

### TestFindBpmPhase
Tests for the internal `_find_bpm_phase` function:
- Basic functionality with simple data
- Forward vs backward phase calculations
- Different target phase values (π/2 vs π)
- Diagonal exclusion (BMPs don't pair with themselves)
- Edge cases (single BPM, two BMPs, empty input)
- Modular arithmetic with phase wrap-around

### Function-Specific Tests
Individual test classes for each public function:
- `TestPrevBpmToPi2`: Tests for `prev_bpm_to_pi_2()`
- `TestNextBpmToPi2`: Tests for `next_bpm_to_pi_2()`
- `TestPrevBpmToPi`: Tests for `prev_bpm_to_pi()`
- `TestNextBpmToPi`: Tests for `next_bpm_to_pi()`

Each includes:
- Basic functionality verification
- Mock testing to ensure correct parameter passing
- Realistic accelerator data scenarios

### Integration Tests
Tests that verify the interaction between different functions:
- Comparing π vs π/2 functions (should give different results)
- Forward vs backward search complementarity
- Consistent tune parameter handling across all functions

### Fixture-Based Tests
Tests using pytest fixtures for realistic scenarios:
- Sample BPM data with known phase patterns
- Realistic accelerator data (100 BMPs with LHC-like distribution)
- Tests with various tune values including realistic LHC tunes

## Running the Tests

```bash
# Run all tests with verbose output
python -m pytest tests/test_bpm_phases.py -v

# Run with coverage
python -m pytest tests/test_bpm_phases.py --cov=src.aba_optimiser.physics.bpm_phases

# Run specific test class
python -m pytest tests/test_bpm_phases.py::TestFindBpmPhase -v
```

## Test Data Patterns

The tests use several data patterns:
- **Simple quadrants**: [0.0, 0.25, 0.5, 0.75] for basic functionality
- **Evenly spaced**: Linear spacing for realistic accelerator scenarios
- **Wrap-around**: [0.9, 0.1, 0.4] to test modular arithmetic
- **Random realistic**: Sorted random phases for statistical validation

## Edge Cases Covered

- Single BPM (should raise ValueError)
- Empty input (should raise ValueError)
- Two BMPs (minimum viable case)
- Non-unit tunes (e.g., 0.5, 62.31)
- Large datasets (100+ BMPs)

All tests pass and provide comprehensive validation of the BPM phase calculation functionality.