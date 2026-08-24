import numpy as np
import pytest

from src.aba_optimiser.physics.deltap import (
    deltap_wrt_reference_total_energy,
    dp2pt,
    get_beam_beta,
    kinetic_to_total_energy,
    momentum_to_total_energy,
)


class TestGetBeamBeta:
    def test_rest_energy(self):
        """Test beta at rest (energy = mass)"""
        mass = 0.938  # proton mass in GeV
        energy = 0.938
        beta = get_beam_beta(mass, energy)
        assert np.isclose(beta, 0.0, rtol=1e-12, atol=1e-15)

    def test_high_energy_limit(self):
        """Test beta approaches 1 at high energy"""
        mass = 0.938
        energy = 1e6  # very high energy
        beta = get_beam_beta(mass, energy)
        assert np.isclose(beta, 1.0, rtol=1e-12, atol=1e-15)

    def test_intermediate_energy(self):
        """Test beta at intermediate energy"""
        mass = 0.938
        energy = 10.0
        beta = get_beam_beta(mass, energy)
        expected = np.sqrt(1 - (mass / energy) ** 2)
        assert np.isclose(beta, expected, rtol=1e-12, atol=1e-15)

    def test_zero_mass(self):
        """Test with zero mass (photons)"""
        mass = 0.0
        energy = 1.0
        beta = get_beam_beta(mass, energy)
        assert np.isclose(beta, 1.0, rtol=1e-12, atol=1e-15)


class TestDp2pt:
    def test_zero_dp(self):
        """Test dp2pt with dp=0 returns 0"""
        dp = 0.0
        mass = 0.938
        energy = 10.0
        pt = dp2pt(dp, mass, energy)
        assert np.isclose(pt, 0.0, rtol=1e-12, atol=1e-15)

    def test_positive_dp(self):
        """Test dp2pt with positive dp"""
        dp = 0.01
        mass = 0.938
        energy = 10.0
        pt = dp2pt(dp, mass, energy)
        # Manual calculation
        beta = get_beam_beta(mass, energy)
        gamma = 1 / beta
        expected = np.sqrt((1 + dp) ** 2 + (gamma**2 - 1)) - gamma
        assert np.isclose(pt, expected, rtol=1e-12, atol=1e-15)

    def test_negative_dp(self):
        """Test dp2pt with negative dp"""
        dp = -0.005
        mass = 0.938
        energy = 10.0
        pt = dp2pt(dp, mass, energy)
        # Manual calculation
        beta = get_beam_beta(mass, energy)
        gamma = 1 / beta
        expected = np.sqrt((1 + dp) ** 2 + (gamma**2 - 1)) - gamma
        assert np.isclose(pt, expected, rtol=1e-12, atol=1e-15)

    def test_high_energy_approximation(self):
        """Test at high energy where pt ≈ dp"""
        dp = 0.01
        mass = 0.938
        energy = 1000.0
        pt = dp2pt(dp, mass, energy)
        # At high energy, pt should be close to dp
        assert np.isclose(pt, dp, rtol=1e-3, atol=1e-6)


class TestMomentumToTotalEnergy:
    def test_zero_momentum_raises(self):
        """Zero momentum is not a valid canonical momentum."""
        with pytest.raises(ValueError):
            momentum_to_total_energy(0.0, 0.938)

    def test_matches_psb_convention(self):
        """Conversion follows sqrt(p^2 + m^2) convention used by PSB."""
        expected = np.sqrt(0.160**2 + 0.93827208816**2)
        assert np.isclose(momentum_to_total_energy(0.160, 0.93827208816), expected)


class TestKineticToTotalEnergy:
    def test_adds_rest_mass(self):
        assert kinetic_to_total_energy(6800.0, 0.93827208816) == pytest.approx(
            6800.93827208816
        )


class TestDeltapWrtReferenceTotalEnergy:
    def test_uses_total_energy_for_reference_offset(self):
        result = deltap_wrt_reference_total_energy(
            kinetic_energy=6800.0,
            machine_deltap=1e-4,
            reference_kinetic_energy=6800.0,
            mass=0.93827208816,
        )

        assert result == pytest.approx(1e-4)

    def test_includes_rest_mass_when_reference_kinetic_energy_differs(self):
        mass = 0.93827208816
        result = deltap_wrt_reference_total_energy(
            kinetic_energy=450.0,
            machine_deltap=0.0,
            reference_kinetic_energy=6800.0,
            mass=mass,
        )
        expected = ((450.0 + mass) - (6800.0 + mass)) / (6800.0 + mass)

        assert result == pytest.approx(expected)
