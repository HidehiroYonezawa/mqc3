"""Tests for nullifiers in MQC3 to verify quantum correlations."""

from math import pi

import numpy as np
import pytest
from allpairspy import AllPairs

from mqc3.client import MQC3ClientResult
from mqc3.constant import hbar
from mqc3.execute import execute
from mqc3.graph import GraphRepr
from mqc3.graph.compose import (
    ComposeSettings,
    MappingInfo,
    compose_into_composite_graph,
)
from mqc3.graph.ops import Measurement
from mqc3.machinery.compose import decompose_composite_machinery_result
from mqc3.machinery.result import MachineryShotMeasuredValue

from .common import construct_client

pytestmark = pytest.mark.network

EXPECTED_SQUEEZING_LEVEL = 5.0
EXPECTED_ANTI_SQUEEZING_LEVEL = 10.0
N_LOCAL_MACRONODES = 101


def run_all_measurement_graph(
    n_shots: int,
    theta: float,
    n_local_macronodes: int,
    composition: bool,  # noqa: FBT001
    max_steps: int,
) -> tuple[MQC3ClientResult, MappingInfo | None, int | None]:
    n_steps = 2
    g = GraphRepr(n_local_macronodes=n_local_macronodes, n_steps=n_steps)
    for k in range(n_local_macronodes * n_steps):
        g.place_operation(Measurement(macronode=g.get_coord(k), theta=theta))

    map_info = None
    n_execs_needed = None
    if composition:
        compose_setting = ComposeSettings(
            n_shots=n_shots,
            n_local_macronodes=N_LOCAL_MACRONODES,
            max_steps=max_steps,
        )
        g_composite, map_info, n_execs_needed = compose_into_composite_graph(
            g,
            compose_setting,
        )
        g = g_composite

    n_shots_effective = n_execs_needed if n_execs_needed is not None else n_shots
    client = construct_client(n_shots=n_shots_effective, backend="emulator")
    result = execute(g, client)

    assert isinstance(result.client_result, MQC3ClientResult)
    return result.client_result, map_info, n_execs_needed


def get_nullifier(n_local_macronodes: int, m: np.ndarray) -> list[np.ndarray]:
    a, b, c, d = 0, 1, 2, 3
    vac = hbar / 2
    m_a = (
        m[a, :, :-1]
        if n_local_macronodes == N_LOCAL_MACRONODES
        else np.delete(m[a, :, :-1], [n_local_macronodes - 1], axis=-1)
    )
    m_b = (
        m[b, :, 1:]
        if n_local_macronodes == N_LOCAL_MACRONODES
        else np.delete(m[b, :, 1:], [n_local_macronodes - 1], axis=-1)
    )
    m_c = m[c, :, :-n_local_macronodes]
    m_d = m[d, :, n_local_macronodes:]
    norm = (2 * vac) ** (-0.5)
    return [(m_a + m_b) * norm, (m_a - m_b) * norm, (m_c + m_d) * norm, (m_c - m_d) * norm]


def calc_nullifier(
    n_shots: int,
    theta: float,
    n_local_macronodes: int,
    composition: bool,  # noqa: FBT001
    max_steps: int,
) -> list[np.ndarray]:
    result, map_info, n_execs_needed = run_all_measurement_graph(
        n_shots, theta, n_local_macronodes, composition, max_steps
    )

    if composition:
        assert map_info
        assert n_execs_needed
        decomposed_machinery_result = decompose_composite_machinery_result(
            result.machinery_result,
            map_info,
        )
        n_frame = len(decomposed_machinery_result.measured_vals)
        first_shot_result = decomposed_machinery_result[0]
        assert isinstance(first_shot_result, MachineryShotMeasuredValue)
        indices = sorted(first_shot_result.index_list())
        m = np.array([
            [[decomposed_machinery_result[frame][index][ch] for index in indices] for frame in range(n_frame)]
            for ch in range(4)
        ])
    else:
        assert result.compiled_machinery
        n_frame = len(result.machinery_result.measured_vals)
        indices = sorted(result.compiled_machinery.readout_macronode_indices)
        m = np.array([
            [[result.machinery_result[frame][index][ch] for index in indices] for frame in range(n_frame)]
            for ch in range(4)
        ])

    return get_nullifier(n_local_macronodes, m)


@pytest.mark.parametrize(
    argnames=("n_local_macronodes"),
    argvalues=[1, 30, 101],
)
def test_nullifier(n_local_macronodes: int):
    n_x = calc_nullifier(1000, pi / 2, n_local_macronodes, composition=False, max_steps=0)
    n_p = calc_nullifier(1000, 0, n_local_macronodes, composition=False, max_steps=0)

    n_sq = [n_p[0], n_x[1], n_p[2], n_x[3]]
    n_asq = [n_x[0], n_p[1], n_x[2], n_p[3]]

    for col in range(4):
        db_sq = 10 * np.log10(np.var(n_sq[col], axis=0))
        db_asq = 10 * np.log10(np.var(n_asq[col], axis=0))
        assert np.all(db_sq < -EXPECTED_SQUEEZING_LEVEL / 2)
        assert np.all(db_sq > -2 * EXPECTED_SQUEEZING_LEVEL)
        assert np.all(db_asq > EXPECTED_ANTI_SQUEEZING_LEVEL / 2)
        assert np.all(db_asq < 2 * EXPECTED_ANTI_SQUEEZING_LEVEL)


@pytest.mark.parametrize(
    argnames=("n_local_macronodes", "max_steps"),
    argvalues=AllPairs([[1, 30, 101], [200, 500]]),
)
def test_nullifier_compose(n_local_macronodes: int, max_steps: int):
    n_x = calc_nullifier(1000, pi / 2, n_local_macronodes, composition=True, max_steps=max_steps)
    n_p = calc_nullifier(1000, 0, n_local_macronodes, composition=True, max_steps=max_steps)

    n_sq = [n_p[0], n_x[1], n_p[2], n_x[3]]
    n_asq = [n_x[0], n_p[1], n_x[2], n_p[3]]

    for col in range(4):
        db_sq = 10 * np.log10(np.var(n_sq[col], axis=0))
        db_asq = 10 * np.log10(np.var(n_asq[col], axis=0))
        assert np.all(db_sq < -EXPECTED_SQUEEZING_LEVEL / 2)
        assert np.all(db_sq > -2 * EXPECTED_SQUEEZING_LEVEL)
        assert np.all(db_asq > EXPECTED_ANTI_SQUEEZING_LEVEL / 2)
        assert np.all(db_asq < 2 * EXPECTED_ANTI_SQUEEZING_LEVEL)
