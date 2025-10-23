"""Test states for modes of quantum circuit."""


# pyright: reportUnusedExpression=false

from math import cos, sin

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mqc3.circuit.program import CircuitRepr
from mqc3.circuit.state import BosonicState, GaussianState


def test_gaussian_state() -> None:
    mean = np.array([1, 2, 3, 4], dtype=np.complex128)
    cov = np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]], dtype=np.float64)
    state = GaussianState(mean=mean, cov=cov)
    assert isinstance(state, GaussianState)
    assert state.n_modes == 2


def test_gaussian_state_error() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        GaussianState(mean=np.array([1, 2, 3], dtype=np.complex128), cov=np.zeros((3, 3), dtype=np.float64))

    with pytest.raises(ValueError):  # noqa: PT011
        GaussianState(mean=np.array([1, 2], dtype=np.complex128), cov=np.zeros((4, 4), dtype=np.float64))


def test_gaussian_state_vacuum() -> None:
    state = GaussianState.vacuum()
    assert isinstance(state, GaussianState)
    assert state.n_modes == 1
    assert np.allclose(state.mean, np.zeros(2, dtype=np.complex128))
    assert np.allclose(state.cov, 0.5 * np.eye(2, dtype=np.float64))


def test_gaussian_state_squeezed() -> None:
    r = np.log(2)
    hbar = 1.0

    # p-squeezed
    phi_90 = np.pi / 2
    state = GaussianState.squeezed(r, phi_90)
    assert isinstance(state, GaussianState)
    assert state.n_modes == 1
    assert np.allclose(state.mean, np.zeros(2, dtype=np.complex128))
    assert np.allclose(state.cov, 0.5 * hbar * np.array([[4, 0], [0, 0.25]], dtype=np.float64))

    # x-squeezed
    phi_0 = 0.0
    state = GaussianState.squeezed(r, phi_0)
    assert np.allclose(state.cov, hbar * 0.5 * np.array([[0.25, 0], [0, 4]], dtype=np.float64))

    # 60 degree squeezing angle
    phi_60 = np.pi / 3
    state = GaussianState.squeezed(r, phi_60)
    rot = np.array([[cos(phi_60), -sin(phi_60)], [sin(phi_60), cos(phi_60)]], dtype=np.float64)
    cov_expected = hbar * 0.5 * rot @ np.array([[0.25, 0], [0, 4]], dtype=np.float64) @ rot.T
    assert np.allclose(state.cov, cov_expected)


def test_gaussian_state_is_vacuum() -> None:
    state = GaussianState.vacuum()
    assert state.is_vacuum()

    mean = np.array([1, 2], dtype=np.complex128)
    cov = np.diag([1, 3]).astype(np.float64)
    non_vacuum = GaussianState(mean, cov)

    assert not non_vacuum.is_vacuum()


def test_bosonic_state() -> None:
    coeffs = np.array([1 / 2] * 2, np.complex128)
    gaussian_states = [
        GaussianState(
            mean=np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
            cov=np.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                ],
                dtype=np.float64,
            ),
        ),
        GaussianState(
            mean=np.array([2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128),
            cov=np.array(
                [
                    [2.0, 3.0],
                    [3.0, 5.0],
                ],
                dtype=np.float64,
            ),
        ),
    ]
    state = BosonicState(coeffs, gaussian_states)

    assert isinstance(state, BosonicState)
    assert state.n_modes == 1
    assert state.n_peaks == 2

    for i in range(state.n_peaks):
        assert state.get_coeff(0) == coeffs[0]
        g_state = state.get_gaussian_state(i)
        assert np.allclose(g_state.mean, gaussian_states[i].mean)
        assert np.allclose(g_state.cov, gaussian_states[i].cov)


def test_bosonic_state_vacuum() -> None:
    """Test vacuum of BosonicState."""
    state = BosonicState.vacuum()

    assert isinstance(state, BosonicState)
    assert state.n_modes == 1
    assert state.n_peaks == 1
    assert np.allclose(state.coeffs, np.array([1.0 + 0.0j], dtype=np.complex128))
    assert np.allclose(state.gaussian_states[0].mean, np.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128))
    assert np.allclose(state.gaussian_states[0].cov, 0.5 * np.eye(2, dtype=np.float64))


def test_bosonic_state_is_vacuum() -> None:
    """Test is_vacuum of BosonicState."""
    state = BosonicState.vacuum()
    assert state.is_vacuum()

    coeffs = np.array([1 / 2] * 2, np.complex128)
    gaussian_states = [
        GaussianState(
            mean=np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
            cov=np.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                ],
                dtype=np.float64,
            ),
        ),
        GaussianState(
            mean=np.array([2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128),
            cov=np.array(
                [
                    [2.0, 3.0],
                    [3.0, 5.0],
                ],
                dtype=np.float64,
            ),
        ),
    ]
    non_vacuum = BosonicState(coeffs, gaussian_states)
    assert not non_vacuum.is_vacuum()


@pytest.mark.parametrize(
    argnames=("x", "p", "parity"), argvalues=[(0, 0, 0.5), (1, 0, 0), (0, 1, 1), (1, 2, 3), (1, 1, 0.75)]
)
def test_bosonic_state_cat(x: float, p: float, parity: float) -> None:
    state = BosonicState.cat(x, p, parity)
    assert state.n_modes == 1
    if np.isclose(x, 0) and np.isclose(p, 0):
        assert state.is_vacuum()
    else:
        assert not state.is_vacuum()
        assert state.n_peaks == 4
        assert np.isclose(np.sum(state.coeffs), 1.0)
        for i in range(state.n_peaks):
            assert np.allclose(state.get_gaussian_state(i).cov, GaussianState.vacuum().cov)
        # coeffs should be integer if parity is an integer.
        if np.isclose(parity, np.round(parity)):
            assert (np.abs(np.imag(state.coeffs)) < 1e-15).all()
        else:
            assert (np.abs(np.imag(state.coeffs)) >= 1e-15).any()


def test_bosonic_state_error() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        BosonicState(
            coeffs=np.array([1 / np.sqrt(3)] * 3, dtype=np.complex128),
            gaussian_states=[GaussianState.vacuum(), GaussianState.vacuum()],
        )

    with pytest.raises(ValueError):  # noqa: PT011
        BosonicState(
            coeffs=np.array([1, 2], dtype=np.complex128),
            gaussian_states=[GaussianState.vacuum(), GaussianState.vacuum()],
        )

    mean = np.array([1, 2, 3, 4], dtype=np.complex128)
    cov = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):  # noqa: PT011
        BosonicState(
            coeffs=np.array([1 / np.sqrt(2)] * 2, dtype=np.complex128),
            gaussian_states=[GaussianState(mean, cov), GaussianState.vacuum()],
        )


def test_get_initial_state() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0)
    circuit.set_initial_state(0, BosonicState.vacuum())

    expected = BosonicState.vacuum()
    actual = circuit.get_initial_state(0)

    assert isinstance(actual, BosonicState)
    assert actual.n_modes == expected.n_modes
    assert np.allclose(actual.coeffs, expected.coeffs)
    assert len(actual.gaussian_states) == 1
    assert np.allclose(actual.gaussian_states[0].mean, expected.gaussian_states[0].mean)
    assert np.allclose(actual.gaussian_states[0].cov, expected.gaussian_states[0].cov)


def test_get_initial_state_error() -> None:
    circuit = CircuitRepr("test")

    with pytest.raises(ValueError):  # noqa: PT011
        circuit.get_initial_state(0)


def test_set_initial_state() -> None:
    circuit = CircuitRepr("test")
    circuit.Q(0)

    mean = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    cov = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )
    gaussian_state = GaussianState(mean, cov)
    new_state = BosonicState(
        np.array([1], dtype=np.complex128),
        [gaussian_state],
    )
    circuit.set_initial_state(0, new_state)
    actual = circuit.get_initial_state(0)

    assert isinstance(actual, BosonicState)
    assert actual.n_modes == new_state.n_modes
    assert np.allclose(actual.coeffs, new_state.coeffs)
    assert len(actual.gaussian_states) == 1
    assert np.allclose(actual.gaussian_states[0].mean, new_state.gaussian_states[0].mean)
    assert np.allclose(actual.gaussian_states[0].cov, new_state.gaussian_states[0].cov)


def test_set_initial_state_error() -> None:
    circuit = CircuitRepr("test")

    with pytest.raises(ValueError):  # noqa: PT011
        circuit.set_initial_state(0, BosonicState.vacuum())

    circuit.Q(0)
    coeffs = np.array([1 / 2] * 2, dtype=np.complex128)
    gaussian_states = [
        GaussianState(
            np.array([1, 0, 0, 1], dtype=np.complex128),
            np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]], dtype=np.float64),
        ),
        GaussianState(
            np.array([1, 2, 3, 4], dtype=np.complex128),
            np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]], dtype=np.float64),
        ),
    ]
    new_state = BosonicState(coeffs, gaussian_states)

    with pytest.raises(ValueError):  # noqa: PT011
        circuit.set_initial_state(0, new_state)


def test_convert_gaussian_state() -> None:
    mean = np.array([1 / np.sqrt(2)] * 2, dtype=np.complex128)
    cov = np.array([[0.5, 0], [0, 0.25]], dtype=np.float64)
    state = GaussianState(mean, cov)

    pb_state = state.proto()
    reconstructed_state = GaussianState.construct_from_proto(pb_state)

    assert isinstance(reconstructed_state, GaussianState)
    assert reconstructed_state.n_modes == state.n_modes
    assert np.allclose(reconstructed_state.mean, state.mean)
    assert np.allclose(reconstructed_state.cov, state.cov)


def test_convert_bosonic_state() -> None:
    mean = np.array([1 / np.sqrt(2)] * 2, dtype=np.complex128)
    cov = np.array([[0.5, 0], [0, 0.25]], dtype=np.float64)
    g_state = GaussianState(mean, cov)

    coeffs = np.array([1], dtype=np.complex128)

    state = BosonicState(coeffs, [g_state])
    pb_state = state.proto()

    reconstructed_state = BosonicState.construct_from_proto(pb_state)

    assert isinstance(reconstructed_state, BosonicState)
    assert reconstructed_state.n_modes == state.n_modes
    assert np.allclose(reconstructed_state.coeffs, state.coeffs)
    assert len(reconstructed_state.gaussian_states) == 1
    assert np.allclose(reconstructed_state.gaussian_states[0].mean, g_state.mean)
    assert np.allclose(reconstructed_state.gaussian_states[0].cov, g_state.cov)


def test_extract_mode() -> None:
    mean0 = np.array([1.0 + 2.0j, -3.5 + 0.5j, 2.2 - 1.1j, 0.0 + 4.0j, -1.0 - 2.0j, 3.3 + 0.0j], dtype=np.complex128)
    cov0 = np.array(
        [
            [6.0, 2.0, 1.0, 0.5, 1.5, 0.0],
            [2.0, 5.0, 0.5, 1.0, 0.0, 1.2],
            [1.0, 0.5, 4.0, 0.8, 0.6, 0.3],
            [0.5, 1.0, 0.8, 3.5, 0.9, 0.7],
            [1.5, 0.0, 0.6, 0.9, 4.5, 1.1],
            [0.0, 1.2, 0.3, 0.7, 1.1, 3.8],
        ],
        dtype=np.float64,
    )
    g_state0 = GaussianState(mean0, cov0)

    mean1 = np.array([0.5 - 1.2j, -2.0 + 3.3j, 4.1 + 0.0j, -0.7 - 0.7j, 1.5 + 2.5j, -3.0 + 1.0j], dtype=np.complex128)
    cov1 = np.array(
        [
            [5.0, -1.0, 0.5, 0.0, -0.3, 0.7],
            [-1.0, 4.5, -0.6, 0.9, 0.2, 0.0],
            [0.5, -0.6, 3.8, 0.4, -0.5, 0.3],
            [0.0, 0.9, 0.4, 4.2, 0.6, -0.8],
            [-0.3, 0.2, -0.5, 0.6, 3.9, 0.4],
            [0.7, 0.0, 0.3, -0.8, 0.4, 4.1],
        ],
        dtype=np.float64,
    )
    g_state1 = GaussianState(mean1, cov1)

    coeffs = np.array([0.7, 0.3], dtype=np.complex128)
    state = BosonicState(coeffs, [g_state0, g_state1])
    extracted_state0 = state.extract_mode(0)
    extracted_state1 = state.extract_mode(1)
    extracted_state2 = state.extract_mode(2)

    assert extracted_state0.get_coeff(0) == 0.7
    assert extracted_state0.get_coeff(1) == 0.3
    assert extracted_state1.get_coeff(0) == 0.7
    assert extracted_state1.get_coeff(1) == 0.3
    assert extracted_state2.get_coeff(0) == 0.7
    assert extracted_state2.get_coeff(1) == 0.3
    with pytest.raises(IndexError):
        extracted_state0.get_coeff(2)
    with pytest.raises(IndexError):
        extracted_state1.get_coeff(2)
    with pytest.raises(IndexError):
        extracted_state2.get_coeff(2)

    assert_array_equal(extracted_state0.get_gaussian_state(0).mean, mean0[[0, 3]])
    assert_array_equal(extracted_state1.get_gaussian_state(0).mean, mean0[[1, 4]])
    assert_array_equal(extracted_state2.get_gaussian_state(0).mean, mean0[[2, 5]])
    assert_array_equal(extracted_state0.get_gaussian_state(1).mean, mean1[[0, 3]])
    assert_array_equal(extracted_state1.get_gaussian_state(1).mean, mean1[[1, 4]])
    assert_array_equal(extracted_state2.get_gaussian_state(1).mean, mean1[[2, 5]])
    with pytest.raises(IndexError):
        extracted_state0.get_gaussian_state(2)
    with pytest.raises(IndexError):
        extracted_state1.get_gaussian_state(2)
    with pytest.raises(IndexError):
        extracted_state2.get_gaussian_state(2)

    assert_array_equal(extracted_state0.get_gaussian_state(0).cov, cov0[np.ix_([0, 3], [0, 3])])
    assert_array_equal(extracted_state1.get_gaussian_state(0).cov, cov0[np.ix_([1, 4], [1, 4])])
    assert_array_equal(extracted_state2.get_gaussian_state(0).cov, cov0[np.ix_([2, 5], [2, 5])])
    assert_array_equal(extracted_state0.get_gaussian_state(1).cov, cov1[np.ix_([0, 3], [0, 3])])
    assert_array_equal(extracted_state1.get_gaussian_state(1).cov, cov1[np.ix_([1, 4], [1, 4])])
    assert_array_equal(extracted_state2.get_gaussian_state(1).cov, cov1[np.ix_([2, 5], [2, 5])])
