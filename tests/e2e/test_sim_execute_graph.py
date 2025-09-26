"""Test the graph simulator (teleportation circuit)."""

from dataclasses import dataclass
from math import cos, log10, pi, sin, tan

import numpy as np
import pytest
from allpairspy import AllPairs
from numpy.typing import NDArray
from scipy.stats import chi2, sem, t

from mqc3.client import SimulatorClient, SimulatorClientResult
from mqc3.constant import hbar
from mqc3.execute import execute
from mqc3.feedforward import feedforward
from mqc3.graph import GraphRepr, GraphResult
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import Initialization, Manual, Measurement, PhaseRotation, Squeezing

pytestmark = pytest.mark.simulator


def calc_measurement_squeezing_level_in_db(res_squeezing_level_in_db: float) -> float:
    return res_squeezing_level_in_db - 10 * np.log10(2)


def calc_squeezing_level_no_unit(squeezing_level_in_db: float) -> float:
    return 10.0 ** (squeezing_level_in_db / 10.0)


def generate_squeezed_cov(squeezing_level: float, phi: float) -> np.ndarray:
    """Generates a 2x2 covariance matrix for a squeezed state.

    Args:
        squeezing_level (float): The squeezing level in dB.
        phi (float): The squeezing angle in radians.

    Returns:
        A tuple representing the 2x2 covariance matrix (a, b, c, d).
    """
    squeezing_level_no_unit = 10 ** (squeezing_level * 0.1)
    squeezing = np.array([
        [1.0 / squeezing_level_no_unit, 0],
        [0, squeezing_level_no_unit],
    ])
    squeezing *= hbar / 2
    rot_mat = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)],
    ])
    return rot_mat @ squeezing @ rot_mat.T


@dataclass
class GaussianDirectionalVariance:
    minor: float
    major: float
    oblique_45: float


def noisy_squeezed_variances(
    squeezing_level: float,
    *,
    n_teleport: int = 1,
    source_cov: NDArray[np.float64] | None = None,
) -> GaussianDirectionalVariance:
    """Returns the expected variances of a measured squeezed state after some teleportations.

    Args:
        squeezing_level (squeezing_level): Squeezing level of the resource squeezed state in dB.
        n_teleport (int): Number of teleportations. Defaults to 1.
        source_cov (NDArray[np.float64] | None): Covariance matrix of the input state of teleportation.
            Defaults to None.

    Returns:
        GaussianDirectionalVariance: An object containing the expected variances of the measured squeezed state.
            - Variance of the minor axis measurement of squeezed states b and d.
            - Variance of the major axis of measurement of squeezed states b and d.
            - Variance of the oblique axis of measurement of squeezed states b and d.
            The measurement basis lies exactly midway between the major and minor axes, rotated 45 degrees from each.
    """
    resource_cov = generate_squeezed_cov(squeezing_level, phi=0)
    u_phi_45 = np.array([
        [np.cos(pi / 4)],
        [np.sin(pi / 4)],
    ])
    resource_minor_variance = resource_cov[0, 0]
    resource_major_variance = resource_cov[1, 1]
    source_minor_variance = resource_minor_variance if source_cov is None else source_cov[0, 0]
    source_major_variance = resource_major_variance if source_cov is None else source_cov[1, 1]
    expected_minor_variance = source_minor_variance + resource_minor_variance * 2 * n_teleport
    expected_major_variance = source_major_variance + resource_minor_variance * 2 * n_teleport
    expected_cov = np.array([
        [expected_minor_variance, 0],
        [0, expected_major_variance],
    ])
    expected_oblique_variance = (u_phi_45.T @ expected_cov @ u_phi_45)[0, 0]
    return GaussianDirectionalVariance(expected_minor_variance, expected_major_variance, expected_oblique_variance)


def get_mean_interval(
    data: list[float],
    significance: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the confidence interval for the mean of one-dimensional data.

    Gaussian distribution is assumed.

    Args:
        data (list[float]): The 1-dimensional data.
        significance (float, optional): Significance of the confidence interval.
            Defaults to 0.05.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of lower bound and upper bound
            of the calculated confidence interval.
    """
    mean = np.mean(data)
    standard_error = sem(data)
    return t.interval(1 - significance, len(data) - 1, loc=mean, scale=standard_error)


def check_mean_value(data: list[float], expected_mean: float, significance: float = 0.05) -> None:
    """Check if the expected mean falls within the confidence interval of the mean for the one-dimensional data.

    Gaussian distribution is assumed.

    Args:
        data (list[float]): The 1-dimensional data.
        expected_mean (float): Expected mean.
        significance (float, optional): Significance of the confidence interval. Defaults to 0.05.
    """
    lower_bound, upper_bound = get_mean_interval(data=data, significance=significance)
    assert lower_bound < expected_mean
    assert expected_mean < upper_bound


def get_variance_interval(
    data: list[float],
    significance: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate the confidence interval for the variance of one-dimensional data.

    Gaussian distribution is assumed.

    Args:
        data (list[float]): The 1-dimensional data.
        significance (float, optional): Significance of the confidence interval.
            Defaults to 0.05.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of lower bound and upper bound
            of the calculated confidence interval.
    """
    s2 = np.var(data, ddof=1)
    dof = len(data) - 1
    lower = dof * s2 / chi2.ppf(1 - significance / 2, dof)
    upper = dof * s2 / chi2.ppf(significance / 2, dof)
    return lower, upper


def check_variance_value(data: list[float], expected_variance: float, significance: float = 0.05) -> None:
    """Check if the expected variance falls within the confidence interval of the variance for the one-dim data.

    Gaussian distribution is assumed.

    Args:
        data (list[float]): The 1-dimensional data.
        expected_variance (float): Expected variance.
        significance (float, optional): Significance of the confidence interval. Defaults to 0.05.
    """
    lower_bound, upper_bound = get_variance_interval(data=data, significance=significance)
    assert lower_bound < expected_variance
    assert expected_variance < upper_bound


def construct_teleportation_graph(
    displacement: tuple[float, float],
    squeezing_theta: float,
    measurement_angle: float,
    phi: float,
) -> GraphRepr:
    @feedforward
    def displace_x(m: float) -> float:
        from math import sqrt  # noqa:PLC0415

        return sqrt(2) * m

    @feedforward
    def displace_p(m: float) -> float:
        from math import sqrt  # noqa:PLC0415

        return -sqrt(2) * m

    graph_repr = GraphRepr(n_local_macronodes=7, n_steps=5)

    # Initialize mode 1 with a p-squeezed state
    graph_repr.place_operation(Initialization(macronode=(0, 1), theta=0, initialized_modes=(BLANK_MODE, 1)))
    # Initialize mode 2 with an x-squeezed state
    graph_repr.place_operation(Initialization(macronode=(1, 0), theta=0.5 * pi, initialized_modes=(BLANK_MODE, 2)))

    # R(0) to mode 1
    graph_repr.place_operation(PhaseRotation(macronode=(0, 2), swap=True, phi=0))
    # R(-pi/2) to mode 2
    graph_repr.place_operation(PhaseRotation(macronode=(1, 1), swap=False, phi=-0.5 * pi))
    # Manual(0, pi/2, pi/4, 3pi/4) to mode 1 and mode 2
    graph_repr.place_operation(
        Manual(
            macronode=(1, 2),
            swap=False,
            theta_a=0,
            theta_b=0.5 * pi,
            theta_c=0.25 * pi,
            theta_d=0.75 * pi,
        ),
    )
    # R(-pi/4) to mode 1
    graph_repr.place_operation(PhaseRotation(macronode=(2, 2), swap=False, phi=-0.25 * pi))
    # R(pi/4) to mode 2
    graph_repr.place_operation(PhaseRotation(macronode=(1, 3), swap=False, phi=0.25 * pi))

    # Initialize mode 0 with a squeezed state with squeezing angle phi
    graph_repr.place_operation(Initialization(macronode=(2, 0), theta=0.5 * pi, initialized_modes=(0, BLANK_MODE)))
    graph_repr.place_operation(Squeezing(macronode=(3, 0), swap=False, theta=squeezing_theta))
    graph_repr.place_operation(
        PhaseRotation(macronode=(4, 0), swap=True, phi=phi + pi / 2),  # +pi/2 is from R(-pi/2) in Squeezing
    )
    # Displacement and R(0) to mode 0
    graph_repr.place_operation(PhaseRotation(macronode=(4, 1), swap=False, phi=0, displacement_k_minus_n=displacement))
    # R(-pi/2) to mode 1
    graph_repr.place_operation(PhaseRotation(macronode=(3, 2), swap=False, phi=-0.5 * pi))
    # Manual(0, pi/2, pi/4, 3pi/4) to mode 0 and mode 1
    graph_repr.place_operation(
        Manual(
            macronode=(4, 2),
            swap=False,
            theta_a=0,
            theta_b=0.5 * pi,
            theta_c=0.25 * pi,
            theta_d=0.75 * pi,
        ),
    )
    # R(-pi/4) to mode 0
    graph_repr.place_operation(PhaseRotation(macronode=(4, 3), swap=True, phi=-0.25 * pi))
    # R(pi/4) to mode 1
    graph_repr.place_operation(PhaseRotation(macronode=(5, 2), swap=False, phi=0.25 * pi))

    # Measure x of mode 0
    graph_repr.place_operation(Measurement(macronode=(5, 3), theta=0.5 * pi))
    x0 = graph_repr.get_mode_measured_value(mode=0)
    # Measure p of mode 1
    graph_repr.place_operation(Measurement(macronode=(6, 2), theta=0.0))
    p1 = graph_repr.get_mode_measured_value(mode=1)
    # Measure mode 2
    graph_repr.place_operation(
        Measurement(
            macronode=(1, 4),
            theta=measurement_angle,
            displacement_k_minus_n=(displace_x(x0), displace_p(p1)),
        ),
    )

    return graph_repr


def simulate_graph(
    graph: GraphRepr,
    n_shots: int,
    resource_squeezing_level: float,
) -> GraphResult:
    client = SimulatorClient(
        n_shots=n_shots, state_save_policy="all", remote=True, resource_squeezing_level=resource_squeezing_level
    )
    result = execute(graph, client)

    assert isinstance(result.client_result, SimulatorClientResult)
    return result.client_result.graph_result


def sample_measured_values(
    n_shots: int,
    resource_squeezing_level: float,
    initial_displacement: tuple[float, float],
    initial_squeezing_theta: float,
    phi: float,
    measurement_angle: float,
) -> list[float]:
    graph = construct_teleportation_graph(
        displacement=initial_displacement,
        squeezing_theta=initial_squeezing_theta,
        measurement_angle=measurement_angle,
        phi=phi,
    )

    result = simulate_graph(graph, n_shots, resource_squeezing_level)
    return [smv[1, 4].m_d for smv in result]


@pytest.mark.parametrize(
    argnames=("x", "p", "phi"),
    argvalues=AllPairs(
        [
            [0.1, -1.0, 5.0, -20.0],
            [0.1, -1.0, 5.0, -20.0],
            [-3 * pi / 4, -pi / 4, 0, pi / 6, pi / 3, pi / 2, pi],
        ],
    ),
)
def test_teleportation_minor_axis(x: float, p: float, phi: float) -> None:
    n_shots = 3000
    res_squeezing_level_in_db = 5
    squeezing_theta = pi / 6
    resource_squeezing_level_in_db = calc_measurement_squeezing_level_in_db(res_squeezing_level_in_db)
    expected_noisy_variance1 = noisy_squeezed_variances(
        resource_squeezing_level_in_db,
        n_teleport=1,
        source_cov=generate_squeezed_cov(
            resource_squeezing_level_in_db - 10 * log10(tan(squeezing_theta) ** 2), phi=0
        ),
    )

    displacement = (x, p)
    measured_values = sample_measured_values(
        n_shots,
        res_squeezing_level_in_db,
        displacement,
        squeezing_theta,
        phi=phi,
        measurement_angle=0.5 * pi - phi,
    )

    significance = 0.0001
    check_mean_value(measured_values, x * cos(-phi) - p * sin(-phi), significance)
    check_variance_value(measured_values, expected_noisy_variance1.minor, significance)


@pytest.mark.parametrize(
    argnames=("x", "p", "phi"),
    argvalues=AllPairs(
        [
            [0.1, -1.0, 5.0, -20.0],
            [0.1, -1.0, 5.0, -20.0],
            [-3 * pi / 4, -pi / 4, 0, pi / 6, pi / 3, pi / 2, pi],
        ],
    ),
)
def test_teleportation_major_axis(x: float, p: float, phi: float) -> None:
    n_shots = 3000
    res_squeezing_level_in_db = 5
    squeezing_theta = pi / 6
    resource_squeezing_level_in_db = calc_measurement_squeezing_level_in_db(res_squeezing_level_in_db)
    expected_noisy_variance1 = noisy_squeezed_variances(
        resource_squeezing_level_in_db,
        n_teleport=1,
        source_cov=generate_squeezed_cov(
            resource_squeezing_level_in_db - 10 * log10(tan(squeezing_theta) ** 2), phi=0
        ),
    )

    displacement = (x, p)
    measured_values = sample_measured_values(
        n_shots,
        res_squeezing_level_in_db,
        displacement,
        squeezing_theta,
        phi=phi,
        measurement_angle=-phi,
    )

    significance = 0.0001
    check_mean_value(measured_values, x * sin(-phi) + p * cos(-phi), significance)
    check_variance_value(measured_values, expected_noisy_variance1.major, significance)


@pytest.mark.parametrize(
    argnames=("x", "p", "phi"),
    argvalues=AllPairs(
        [
            [0.1, -1.0, 5.0, -20.0],
            [0.1, -1.0, 5.0, -20.0],
            [-3 * pi / 4, -pi / 4, 0, pi / 6, pi / 3, pi / 2, pi],
        ],
    ),
)
def test_teleportation_oblique_axis(x: float, p: float, phi: float) -> None:
    n_shots = 3000
    res_squeezing_level_in_db = 5
    squeezing_theta = pi / 6
    resource_squeezing_level_in_db = calc_measurement_squeezing_level_in_db(res_squeezing_level_in_db)
    expected_noisy_variance1 = noisy_squeezed_variances(
        resource_squeezing_level_in_db,
        n_teleport=1,
        source_cov=generate_squeezed_cov(
            resource_squeezing_level_in_db - 10 * log10(tan(squeezing_theta) ** 2), phi=0
        ),
    )

    displacement = (x, p)
    measured_values = sample_measured_values(
        n_shots,
        res_squeezing_level_in_db,
        displacement,
        squeezing_theta,
        phi=phi,
        measurement_angle=pi / 4 - phi,
    )

    significance = 0.0001
    check_mean_value(measured_values, x * cos(-phi - pi / 4) - p * sin(-phi - pi / 4), significance)
    check_variance_value(measured_values, expected_noisy_variance1.oblique_45, significance)
