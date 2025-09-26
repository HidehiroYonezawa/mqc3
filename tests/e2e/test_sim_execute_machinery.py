"""Test nullifier to check quantum correlation."""

from math import pi

import numpy as np
import pytest

from mqc3.client import SimulatorClient, SimulatorClientResult
from mqc3.execute import execute
from mqc3.graph import GraphRepr
from mqc3.graph.ops import Measurement
from mqc3.machinery import MachineryRepr

pytestmark = pytest.mark.simulator


def calc_nullifier(  # noqa: PLR0914
    n_shots: int,
    squeezing_level: float,
    theta: float,
    n_local_macronodes: int,
) -> list[np.ndarray]:
    n_steps = 2
    graph_repr = GraphRepr(n_local_macronodes=n_local_macronodes, n_steps=n_steps)
    for k in range(n_local_macronodes * n_steps):
        graph_repr.place_operation(Measurement(macronode=graph_repr.get_coord(k), theta=theta))

    machinery_repr = MachineryRepr.from_graph_repr(graph_repr)

    client = SimulatorClient(
        n_shots=n_shots, state_save_policy="all", remote=True, resource_squeezing_level=squeezing_level
    )
    result = execute(machinery_repr, client)

    assert isinstance(result.client_result, SimulatorClientResult)
    machinery_result = result.client_result.machinery_result

    n_frame = len(machinery_result.measured_vals)
    indices = sorted(machinery_repr.readout_macronode_indices)
    m = np.array([
        [[machinery_result[frame][index][ch] for index in indices] for frame in range(n_frame)] for ch in range(4)
    ])

    a, b, c, d = 0, 1, 2, 3
    hbar = 1
    vac = hbar / 2
    m_a = m[a, :, :-1]
    m_b = m[b, :, 1:]
    m_c = m[c, :, :-n_local_macronodes]
    m_d = m[d, :, n_local_macronodes:]
    norm = (2 * vac) ** (-0.5)
    return [(m_a + m_b) * norm, (m_a - m_b) * norm, (m_c + m_d) * norm, (m_c - m_d) * norm]


@pytest.mark.parametrize(
    "n_local_macronodes",
    [1, 2, 3, 4, 5],
)
def test_nullifier(n_local_macronodes: int):
    n_shots = 1000
    squeezing_level = 15
    n_x = calc_nullifier(
        n_shots=n_shots,
        squeezing_level=squeezing_level,
        theta=pi / 2,
        n_local_macronodes=n_local_macronodes,
    )
    n_p = calc_nullifier(
        n_shots=n_shots,
        squeezing_level=squeezing_level,
        theta=0,
        n_local_macronodes=n_local_macronodes,
    )

    n_sq = [n_p[0], n_x[1], n_p[2], n_x[3]]
    n_asq = [n_x[0], n_p[1], n_x[2], n_p[3]]

    for col in range(4):
        assert np.all(10 * np.log10(np.var(n_sq[col], axis=0)) < 0)
        assert np.all(10 * np.log10(np.var(n_asq[col], axis=0)) > 0)
