"""Tests for executing circuit representations on the local simulator."""

# pyright: reportUnusedExpression=false

from datetime import timedelta
from math import acos, cos, exp, pi, sin, sqrt, tan

import numpy as np
import pytest
from allpairspy import AllPairs

pytest.importorskip("strawberryfields")


from mqc3.circuit import BosonicState, CircuitRepr, GaussianState
from mqc3.circuit.ops import intrinsic
from mqc3.client import SimulatorClient, SimulatorClientResult
from mqc3.constant import hbar
from mqc3.execute import execute

from .common import teleportation_circuit


def _get_state_after_simulation(circuit: CircuitRepr, *, n_modes: int) -> BosonicState:
    client = SimulatorClient(n_shots=1, state_save_policy="all", remote=False)
    result = execute(circuit, client)

    assert isinstance(result.client_result, SimulatorClientResult)
    assert result.execution_time >= timedelta(seconds=0)
    assert result.n_shots == client.n_shots

    assert len(result.client_result.states) == client.n_shots
    assert result.client_result.execution_time is not None
    assert result.client_result.execution_time >= timedelta(seconds=0)
    assert result.client_result.circuit_result.n_shots() == client.n_shots

    state = result.client_result.states[0]
    assert state.n_modes == n_modes
    assert len(state.coeffs) == 1
    assert len(state.gaussian_states) == 1
    assert state.get_coeff(0) == 1.0 + 0.0j

    return state


def test_measurement():
    circuit = CircuitRepr("measurement")
    circuit.Q(0) | intrinsic.Displacement(1.0, 0.0)
    circuit.Q(0) | intrinsic.Measurement(theta=1.0)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))

    gaussian = _get_state_after_simulation(circuit, n_modes=1).get_gaussian_state(0)
    assert gaussian.mean.shape == (2,)
    assert gaussian.cov.shape == (2, 2)
    assert np.allclose(gaussian.mean, GaussianState.vacuum().mean)
    assert np.allclose(gaussian.cov, GaussianState.vacuum().cov)


def test_displacement():
    circuit = CircuitRepr("displacement")
    circuit.Q(0) | intrinsic.Displacement(1.0, 0.0)
    circuit.set_initial_state(0, BosonicState.vacuum())

    gaussian = _get_state_after_simulation(circuit, n_modes=1).get_gaussian_state(0)
    assert gaussian.mean.shape == (2,)
    assert gaussian.cov.shape == (2, 2)
    assert np.allclose(gaussian.mean, (1.0 + 0.0j, 0.0 + 0.0j))
    assert np.allclose(gaussian.cov, GaussianState.vacuum().cov)


def test_phase_rotation():
    circuit = CircuitRepr("phase_rotation")
    phi = 1.0
    circuit.Q(0) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(0) | intrinsic.PhaseRotation(phi=phi)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))

    gaussian = _get_state_after_simulation(circuit, n_modes=1).get_gaussian_state(0)
    assert gaussian.mean.shape == (2,)
    assert gaussian.cov.shape == (2, 2)
    rot = np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])
    assert np.allclose(gaussian.mean, rot @ np.array([1.0, 1.0]))
    assert np.allclose(gaussian.cov, rot @ GaussianState.squeezed(r=1.0, phi=1.0).cov @ rot.T)


@pytest.mark.parametrize("kappa", range(-10, 10))
def test_shear_x_invariant(kappa: float):
    circuit = CircuitRepr("shear_x_invariant")
    circuit.Q(0) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(0) | intrinsic.ShearXInvariant(kappa=kappa)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))

    gaussian = _get_state_after_simulation(circuit, n_modes=1).get_gaussian_state(0)
    assert gaussian.mean.shape == (2,)
    assert gaussian.cov.shape == (2, 2)
    p_matrix = np.array([[1, 0], [2 * kappa, 1]])
    assert np.allclose(gaussian.mean, p_matrix @ np.array([1.0, 1.0]))
    assert np.allclose(gaussian.cov, p_matrix @ GaussianState.squeezed(r=1.0, phi=1.0).cov @ p_matrix.T)


@pytest.mark.parametrize("eta", range(-10, 10))
def test_shear_p_invariant(eta: float):
    circuit = CircuitRepr("shear_q_invariant")
    circuit.Q(0) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(0) | intrinsic.ShearPInvariant(eta=eta)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))

    gaussian = _get_state_after_simulation(circuit, n_modes=1).get_gaussian_state(0)
    assert gaussian.mean.shape == (2,)
    assert gaussian.cov.shape == (2, 2)
    q_matrix = np.array([[1, 2 * eta], [0, 1]])
    assert np.allclose(gaussian.mean, q_matrix @ np.array([1.0, 1.0]))
    assert np.allclose(gaussian.cov, q_matrix @ GaussianState.squeezed(r=1.0, phi=1.0).cov @ q_matrix.T)


@pytest.mark.parametrize("theta", np.linspace(-2 * pi, 2 * pi, 33))
def test_squeezing(theta: float):
    if theta % (pi / 2) == pytest.approx(0.0):
        return
    circuit = CircuitRepr("squeezing")
    circuit.Q(0) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(0) | intrinsic.Squeezing(theta=theta)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))

    gaussian = _get_state_after_simulation(circuit, n_modes=1).get_gaussian_state(0)
    assert gaussian.mean.shape == (2,)
    assert gaussian.cov.shape == (2, 2)
    s_matrix = np.array([[0, 1], [-1, 0]]) @ np.array([[1.0 / tan(theta), 0], [0, tan(theta)]])
    assert np.allclose(gaussian.mean, s_matrix @ np.array([1.0, 1.0]))
    assert np.allclose(gaussian.cov, s_matrix @ GaussianState.squeezed(r=1.0, phi=1.0).cov @ s_matrix.T)


@pytest.mark.parametrize("theta", np.linspace(-2 * pi, 2 * pi, 33))
def test_squeezing_45(theta: float):
    if theta % (pi / 2) == pytest.approx(0.0):
        return
    circuit = CircuitRepr("squeezing_45")
    circuit.Q(0) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(0) | intrinsic.Squeezing45(theta=theta)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))

    gaussian = _get_state_after_simulation(circuit, n_modes=1).get_gaussian_state(0)
    assert gaussian.mean.shape == (2,)
    assert gaussian.cov.shape == (2, 2)
    coeff = 1.0 / sqrt(2)
    s_matrix = (
        np.array([[coeff, coeff], [-coeff, coeff]])
        @ np.array([
            [1.0 / tan(theta), 0],
            [0, tan(theta)],
        ])
        @ np.array([[coeff, -coeff], [coeff, coeff]])
    )
    assert np.allclose(gaussian.mean, s_matrix @ np.array([1.0, 1.0]))
    assert np.allclose(gaussian.cov, s_matrix @ GaussianState.squeezed(r=1.0, phi=1.0).cov @ s_matrix.T)


@pytest.mark.parametrize(
    argnames=("alpha", "beta", "lam"),
    argvalues=AllPairs([
        list(np.linspace(-pi, pi, 17)),
        list(np.linspace(-pi, pi, 17)),
        range(-10, 10),
    ]),
)
def test_arbitrary(alpha: float, beta: float, lam: float):
    circuit = CircuitRepr("arbitrary")
    circuit.Q(0) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(0) | intrinsic.Arbitrary(alpha=alpha, beta=beta, lam=lam)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))

    gaussian = _get_state_after_simulation(circuit, n_modes=1).get_gaussian_state(0)
    assert gaussian.mean.shape == (2,)
    assert gaussian.cov.shape == (2, 2)
    a = (
        np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
        @ np.array([
            [exp(-lam), 0],
            [0, exp(lam)],
        ])
        @ np.array([[cos(beta), -sin(beta)], [sin(beta), cos(beta)]])
    )
    assert np.allclose(gaussian.mean, a @ np.array([1.0, 1.0]))
    assert np.allclose(gaussian.cov, a @ GaussianState.squeezed(r=1.0, phi=1.0).cov @ a.T)


@pytest.mark.parametrize("g", range(-10, 10))
def test_controlled_z(g: float):
    circuit = CircuitRepr("controlled_z")
    circuit.Q(0) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(1) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(0, 1) | intrinsic.ControlledZ(g=g)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))
    circuit.set_initial_state(1, BosonicState.squeezed(r=2.0, phi=3.0))
    sq_0 = GaussianState.squeezed(r=1.0, phi=1.0).cov
    sq_1 = GaussianState.squeezed(r=2.0, phi=3.0).cov

    gaussian = _get_state_after_simulation(circuit, n_modes=2).get_gaussian_state(0)
    assert gaussian.mean.shape == (4,)
    assert gaussian.cov.shape == (4, 4)
    s_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, g, 1, 0], [g, 0, 0, 1]])
    assert np.allclose(gaussian.mean, s_matrix @ np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.allclose(
        gaussian.cov,
        s_matrix
        @ np.array([
            [sq_0[0, 0], 0, sq_0[0, 1], 0],
            [0, sq_1[0, 0], 0, sq_1[0, 1]],
            [sq_0[1, 0], 0, sq_0[1, 1], 0],
            [0, sq_1[1, 0], 0, sq_1[1, 1]],
        ])
        @ s_matrix.T,
    )


@pytest.mark.parametrize(
    argnames=("sqrt_r", "theta_rel"),
    argvalues=AllPairs([
        list(np.linspace(0, 1, 11)),
        list(np.linspace(-2 * pi, 2 * pi, 17)),
    ]),
)
def test_beam_splitter(sqrt_r: float, theta_rel: float):
    circuit = CircuitRepr("beam_splitter")
    circuit.Q(0) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(1) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(0, 1) | intrinsic.BeamSplitter(sqrt_r=sqrt_r, theta_rel=theta_rel)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))
    circuit.set_initial_state(1, BosonicState.squeezed(r=2.0, phi=3.0))
    sq_0 = GaussianState.squeezed(r=1.0, phi=1.0).cov
    sq_1 = GaussianState.squeezed(r=2.0, phi=3.0).cov

    gaussian = _get_state_after_simulation(circuit, n_modes=2).get_gaussian_state(0)
    assert gaussian.mean.shape == (4,)
    assert gaussian.cov.shape == (4, 4)
    alpha = (theta_rel + acos(sqrt_r)) / 2
    beta = (theta_rel - acos(sqrt_r)) / 2
    plus = alpha + beta
    minus = alpha - beta
    cp, cm, sp, sm = cos(plus), cos(minus), sin(plus), sin(minus)
    b = np.array([
        [cp * cm, sp * sm, -sp * cm, sm * cp],
        [sp * sm, cp * cm, sm * cp, -sp * cm],
        [sp * cm, -sm * cp, cp * cm, sp * sm],
        [-sm * cp, sp * cm, sp * sm, cp * cm],
    ])
    assert np.allclose(gaussian.mean, b @ np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.allclose(
        gaussian.cov,
        b
        @ np.array([
            [sq_0[0, 0], 0, sq_0[0, 1], 0],
            [0, sq_1[0, 0], 0, sq_1[0, 1]],
            [sq_0[1, 0], 0, sq_0[1, 1], 0],
            [0, sq_1[1, 0], 0, sq_1[1, 1]],
        ])
        @ b.T,
    )


@pytest.mark.parametrize(
    argnames=("a", "b"),
    argvalues=AllPairs([
        list(np.linspace(0, 1, 11)),
        list(np.linspace(0, 1, 11)),
    ]),
)
def test_two_mode_shear(a: float, b: float):
    circuit = CircuitRepr("two_mode_shear")
    circuit.Q(0) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(1) | intrinsic.Displacement(1.0, 1.0)
    circuit.Q(0, 1) | intrinsic.TwoModeShear(a=a, b=b)
    circuit.set_initial_state(0, BosonicState.squeezed(r=1.0, phi=1.0))
    circuit.set_initial_state(1, BosonicState.squeezed(r=2.0, phi=3.0))
    sq_0 = GaussianState.squeezed(r=1.0, phi=1.0).cov
    sq_1 = GaussianState.squeezed(r=2.0, phi=3.0).cov

    gaussian = _get_state_after_simulation(circuit, n_modes=2).get_gaussian_state(0)
    assert gaussian.mean.shape == (4,)
    assert gaussian.cov.shape == (4, 4)
    p2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [2 * a, b, 1, 0], [b, 2 * a, 0, 1]])
    assert np.allclose(gaussian.mean, p2 @ np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.allclose(
        gaussian.cov,
        p2
        @ np.array([
            [sq_0[0, 0], 0, sq_0[0, 1], 0],
            [0, sq_1[0, 0], 0, sq_1[0, 1]],
            [sq_0[1, 0], 0, sq_0[1, 1], 0],
            [0, sq_1[1, 0], 0, sq_1[1, 1]],
        ])
        @ p2.T,
    )


def test_teleportation():
    disp_x = 15.0
    disp_p = -2.78
    sq = 1.56
    circuit = teleportation_circuit((disp_x, disp_p), sq)

    client = SimulatorClient(n_shots=5, state_save_policy="all", remote=False)
    result = execute(circuit, client)

    assert isinstance(result.client_result, SimulatorClientResult)
    assert result.execution_time >= timedelta(seconds=0)
    assert result.n_shots == client.n_shots

    assert len(result.client_result.states) == client.n_shots
    assert result.client_result.execution_time is not None
    assert result.client_result.execution_time >= timedelta(seconds=0)
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

        mean = gaussian.mean
        assert mean[2] == pytest.approx(disp_x, rel=1e-2)
        assert mean[5] == pytest.approx(disp_p, rel=1e-2)

        cov = gaussian.cov
        for i in [0, 1, 3, 4]:
            assert cov[i][i] == pytest.approx(hbar / 2, rel=1e-5)
        assert cov[2][2] == pytest.approx(10**sq * hbar / 2, rel=1e-5)
        assert cov[2][5] == pytest.approx(0, abs=1e-5)
        assert cov[5][2] == pytest.approx(0, abs=1e-5)
        assert cov[5][5] == pytest.approx(10 ** (-sq) * hbar / 2, rel=1e-5)
