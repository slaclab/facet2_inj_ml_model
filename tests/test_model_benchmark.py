"""Benchmark / regression tests for the FACET-II injector ML model.

These tests ensure the model continues to produce the same outputs over time.
Run with:
    pytest tests/test_model_benchmark.py -v
"""

import numpy as np
import pytest
import torch

from facet2_inj_ml_model import load_model
from facet2_inj_ml_model.beam_output_model import BeamOutputModel
from facet2_inj_ml_model.pv_mapping import (
    machine_input_names,
    machine_to_sim_array,
    sim_to_machine_array,
)

# ---------------------------------------------------------------------------
# Reference input (machine PV units)
# ---------------------------------------------------------------------------
EXAMPLE_INPUT = {
    "QUAD:IN10:121:BCTRL": 0.03196033090353012,
    "KLYS:LI10:21:AMPL": 40.167457580566406,
    "KLYS:LI10:21:PHAS": 74.87874603271484,
    "SOLN:IN10:121:BCTRL": 0.43,
    "QUAD:IN10:122:BCTRL": 0.0315588042140007,
    "distgen:t_dist:sigma_t:value": 0.43811073899269104,
    "TORO:IN10:591:TMIT_PC": 954.0297241210938,
}

# Reference covariance matrix from verified model run
EXPECTED_COVARIANCE = torch.tensor([
    [ 3.4176e-07,  2.3318e+00, -4.6129e-08, -3.5479e-01, -1.7159e-16,  5.4710e+00],
    [ 2.3318e+00,  1.0546e+08, -4.8810e-01, -8.8954e+06, -4.2216e-09,  9.9981e+07],
    [-4.6129e-08, -4.8810e-01,  1.9479e-07,  7.9473e-01,  1.3557e-16, -5.4903e+00],
    [-3.5479e-01, -8.8954e+06,  7.9473e-01,  1.0056e+08,  3.3319e-09, -1.0944e+08],
    [-1.7159e-16, -4.2216e-09,  1.3557e-16,  3.3319e-09,  2.6001e-24, -1.6487e-07],
    [ 5.4710e+00,  9.9981e+07, -5.4903e+00, -1.0944e+08, -1.6487e-07,  1.4602e+10],
])

# Sim-space input variables (must match model training order)
SIM_INPUT_VARIABLES = [
    "CQ10121:b1_gradient",
    "GUNF:rf_field_scale",
    "GUNF:theta0_deg",
    "SOL10111:solenoid_field_scale",
    "SQ10122:b1_gradient",
    "distgen:t_dist:sigma_t:value",
    "distgen:total_charge:value",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def machine_model():
    """Load the machine-PV model once for the entire test module."""
    return load_model("machine")


@pytest.fixture(scope="module")
def sim_model():
    """Load the sim model once for the entire test module."""
    return load_model("sim")


@pytest.fixture(scope="module")
def beam_model():
    """Create a BeamOutputModel wrapping the machine model."""
    return BeamOutputModel(load_model("machine"), n_particles=10000)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestModelLoading:
    """Verify that both model variants load without errors."""

    def test_load_machine_model(self, machine_model):
        assert machine_model is not None

    def test_load_sim_model(self, sim_model):
        assert sim_model is not None

    def test_machine_model_has_output_variables(self, machine_model):
        var_names = [v.name for v in machine_model.output_variables]
        assert "covariance_matrix" in var_names

    def test_sim_model_has_output_variables(self, sim_model):
        var_names = [v.name for v in sim_model.output_variables]
        assert "covariance_matrix" in var_names


class TestEvaluate:
    """Verify model.evaluate() produces expected outputs."""

    def test_evaluate_returns_covariance_key(self, machine_model):
        result = machine_model.evaluate(EXAMPLE_INPUT)
        assert "covariance_matrix" in result

    def test_covariance_shape(self, machine_model):
        result = machine_model.evaluate(EXAMPLE_INPUT)
        cov = result["covariance_matrix"]
        assert cov.shape == (6, 6)

    def test_covariance_is_symmetric(self, machine_model):
        result = machine_model.evaluate(EXAMPLE_INPUT)
        cov = result["covariance_matrix"]
        torch.testing.assert_close(cov, cov.T, atol=1e-6, rtol=1e-5)

    def test_covariance_eigenvalues_reasonable(self, machine_model):
        """Check eigenvalues are not wildly negative (small negatives are acceptable numerical noise)."""
        result = machine_model.evaluate(EXAMPLE_INPUT)
        cov = result["covariance_matrix"]
        eigenvalues = torch.linalg.eigvalsh(cov)
        max_eig = eigenvalues.abs().max()
        assert (eigenvalues >= -1e-4 * max_eig).all(), f"Large negative eigenvalue found: {eigenvalues}"

    def test_covariance_matches_reference(self, machine_model):
        """Regression test: covariance must match the verified reference output."""
        result = machine_model.evaluate(EXAMPLE_INPUT)
        cov = result["covariance_matrix"]
        torch.testing.assert_close(cov, EXPECTED_COVARIANCE, atol=1e-6, rtol=1e-3)


class TestBeamOutputModel:
    """Verify the BeamOutputModel wrapper produces a valid particle distribution."""

    def test_beam_model_produces_particles(self, beam_model):
        beam_model.set(EXAMPLE_INPUT)
        output_beam = beam_model.final_particles
        assert output_beam is not None
        assert len(output_beam.x) == 10000

    def test_beam_particle_dimensions(self, beam_model):
        beam_model.set(EXAMPLE_INPUT)
        output_beam = beam_model.final_particles
        assert len(output_beam.x) == len(output_beam.px)
        assert len(output_beam.y) == len(output_beam.py)
        assert len(output_beam.t) == len(output_beam.pz)

    def test_beam_mean_near_zero(self, beam_model):
        """Particle distribution should be zero-mean (sampled from zero-mean MVN)."""
        beam_model.set(EXAMPLE_INPUT)
        output_beam = beam_model.final_particles
        assert abs(output_beam.x.mean()) < 5 * output_beam.x.std() / np.sqrt(len(output_beam.x))
        assert abs(output_beam.y.mean()) < 5 * output_beam.y.std() / np.sqrt(len(output_beam.y))

    def test_beam_std_x_in_expected_range(self, beam_model):
        """Std of x should be consistent with the covariance diagonal."""
        beam_model.set(EXAMPLE_INPUT)
        output_beam = beam_model.final_particles
        expected_std_x = np.sqrt(EXPECTED_COVARIANCE[0, 0].item())
        actual_std_x = output_beam.x.std()
        assert abs(actual_std_x - expected_std_x) / expected_std_x < 0.2


class TestPVMapping:
    """Verify PV ↔ sim parameter conversions are consistent."""

    def test_machine_input_names_length(self):
        names = machine_input_names(SIM_INPUT_VARIABLES)
        assert len(names) == len(SIM_INPUT_VARIABLES)

    def test_roundtrip_conversion(self):
        """sim → machine → sim should recover original values."""
        sim_values = np.array([0.01, 3.2e-5, -77.5, 0.27, -0.015, 0.44, 950.0], dtype=np.float32)
        machine_values = sim_to_machine_array(sim_values, SIM_INPUT_VARIABLES)
        recovered = machine_to_sim_array(machine_values, SIM_INPUT_VARIABLES)
        np.testing.assert_allclose(recovered, sim_values, rtol=1e-4)
