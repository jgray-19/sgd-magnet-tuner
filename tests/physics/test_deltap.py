import numpy as np

from src.aba_optimiser.physics.deltap import dp2pt, get_beam_beta


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
