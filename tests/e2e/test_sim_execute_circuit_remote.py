"""Tests for executing circuit representations on the remote simulator."""

# pyright: reportUnusedExpression=false

from datetime import timedelta

import pytest

from mqc3.circuit import BosonicState, CircuitRepr, GaussianState
from mqc3.circuit.ops import intrinsic
from mqc3.circuit.result import CircuitShotMeasuredValue
from mqc3.client import SimulatorClient, SimulatorClientResult
from mqc3.client.simulator_client import BackendType, StateSavePolicy
from mqc3.constant import hbar
from mqc3.execute import execute

from .common import teleportation_circuit

pytestmark = pytest.mark.simulator


def test_displacement():
    circuit = CircuitRepr("displacement")
    circuit.Q(0) | intrinsic.Displacement(1.0, 0.0)
    circuit.set_initial_state(0, BosonicState.vacuum())

    client = SimulatorClient(n_shots=1, state_save_policy="all", remote=True)
    result = execute(circuit, client)

    assert isinstance(result.client_result, SimulatorClientResult)
    assert result.execution_time >= timedelta(seconds=0)
    assert result.n_shots == client.n_shots

    assert len(result.client_result.states) == client.n_shots
    assert result.client_result.execution_time
    assert result.client_result.execution_time >= timedelta(seconds=0)
    assert result.client_result.circuit_result.n_shots() == client.n_shots

    state = result.client_result.states[0]
    assert state.n_modes == 1
    assert len(state.coeffs) == 1
    assert len(state.gaussian_states) == 1
    assert state.get_coeff(0) == 1.0 + 0.0j

    gaussian = state.get_gaussian_state(0)
    assert gaussian.mean.shape == (2,)
    assert gaussian.cov.shape == (2, 2)
    assert (gaussian.mean == (1.0 + 0.0j, 0.0 + 0.0j)).all()
    assert (gaussian.cov == GaussianState.vacuum().cov).all()


@pytest.mark.parametrize(
    argnames=("backend"),
    argvalues=["cpu", "single_gpu", "multi_gpu"],
)
def test_teleportation(backend: BackendType):
    disp_x = 15.0
    disp_p = -2.7
    sq = 1.56
    circuit = teleportation_circuit((disp_x, disp_p), sq)

    client = SimulatorClient(n_shots=5, backend=backend, state_save_policy="all", remote=True)
    result = execute(circuit, client)

    assert isinstance(result.client_result, SimulatorClientResult)
    assert result.execution_time > timedelta(seconds=0)
    assert result.n_shots == client.n_shots

    assert len(result.client_result.states) == client.n_shots
    assert result.client_result.execution_time
    assert result.client_result.execution_time > timedelta(seconds=0)
    assert result.client_result.circuit_result.n_shots() == client.n_shots

    state = result.client_result.states[0]
    assert state.n_modes == 3
    assert len(state.coeffs) == 1
    assert len(state.gaussian_states) == 1

    gaussian = state.get_gaussian_state(0)
    assert gaussian.mean.shape == (3 * 2,)
    assert gaussian.cov.shape == (3 * 2, 3 * 2)

    for shot in range(client.n_shots):
        state = result.client_result.states[shot]
        gaussian = state.get_gaussian_state(0)
        measured = result.client_result.circuit_result.measured_vals[shot]
        assert isinstance(measured, CircuitShotMeasuredValue)

        mean = gaussian.mean
        m0 = measured.get_value(0)
        m1 = measured.get_value(1)
        assert m0 is not None
        assert m1 is not None
        assert mean[2] == pytest.approx(disp_x, rel=1e-2)
        assert mean[5] == pytest.approx(disp_p, rel=1e-2)

        cov = gaussian.cov
        for i in [0, 1, 3, 4]:
            assert cov[i][i] == pytest.approx(hbar / 2, rel=1e-5)
        assert cov[2][2] == pytest.approx(10**sq * hbar / 2, rel=1e-5)
        assert cov[2][5] == pytest.approx(0, abs=1e-5)
        assert cov[5][2] == pytest.approx(0, abs=1e-5)
        assert cov[5][5] == pytest.approx(10 ** (-sq) * hbar / 2, rel=1e-5)


@pytest.mark.parametrize(
    argnames=("state_save_policy"),
    argvalues=["all", "first_only", "none"],
)
def test_state_save_policy(state_save_policy: StateSavePolicy):
    circuit = teleportation_circuit((1.0, 0.5), 10.0)

    client = SimulatorClient(n_shots=100, state_save_policy=state_save_policy, remote=True)
    result = execute(circuit, client)

    assert isinstance(result.client_result, SimulatorClientResult)
    assert result.execution_time > timedelta(seconds=0)
    assert result.n_shots == client.n_shots

    if state_save_policy == "all":
        assert len(result.client_result.states) == client.n_shots
        assert result.client_result.execution_time
        assert result.client_result.execution_time > timedelta(seconds=0)
        assert result.client_result.execution_details.timeline is not None
        assert result.client_result.execution_details.timeline.total_time is not None
        assert result.client_result.execution_details.timeline.total_time > timedelta(seconds=0)
        assert result.client_result.circuit_result.n_shots() == client.n_shots

        state = result.client_result.states[0]
        assert state.n_modes == 3
        assert len(state.coeffs) == 1
        assert len(state.gaussian_states) == 1

        gaussian = state.gaussian_states[0]
        assert len(gaussian.mean) == 3 * 2
        assert len(gaussian.cov) == 3 * 2
    elif state_save_policy == "first_only":
        assert len(result.client_result.states) == 1
        assert result.client_result.execution_time
        assert result.client_result.execution_time > timedelta(seconds=0)
        assert result.client_result.execution_details.timeline is not None
        assert result.client_result.execution_details.timeline.total_time is not None
        assert result.client_result.execution_details.timeline.total_time > timedelta(seconds=0)
        assert result.client_result.circuit_result.n_shots() == client.n_shots

        state = result.client_result.states[0]
        assert state.n_modes == 3
        assert len(state.coeffs) == 1
        assert len(state.gaussian_states) == 1

        gaussian = state.gaussian_states[0]
        assert len(gaussian.mean) == 3 * 2
        assert len(gaussian.cov) == 3 * 2
    elif state_save_policy == "none":
        assert len(result.client_result.states) == 0
        assert result.client_result.execution_time
        assert result.client_result.execution_time > timedelta(seconds=0)
        assert result.client_result.execution_details.timeline is not None
        assert result.client_result.execution_details.timeline.total_time is not None
        assert result.client_result.execution_details.timeline.total_time > timedelta(seconds=0)
        assert result.client_result.circuit_result.n_shots() == client.n_shots
