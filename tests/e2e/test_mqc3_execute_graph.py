"""Tests for executing graph representations using an MQC3 client."""

from datetime import timedelta

import numpy as np
import pytest
from numpy import pi

import mqc3.graph.ops as gops
from mqc3.client import MQC3ClientResult
from mqc3.execute import execute
from mqc3.feedforward import feedforward
from mqc3.graph import GraphRepr, GraphResult
from mqc3.graph.compose import ComposeSettings, compose_into_composite_graph, decompose_composite_graph_result
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    ControlledZ,
    Initialization,
    Measurement,
    PhaseRotation,
)
from mqc3.graph.visualize import savefig

from .common import construct_client

pytestmark = pytest.mark.network

plot_on = False


def test_execute_graph_with_ff():
    """Graph with nlff.

      0     1  2    3  4
    0          I(1)
    1 I(0)  R  C    R  M(1)
    2          M(0)
    """
    g = GraphRepr(3, 5)

    g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(PhaseRotation((1, 1), pi / 2, swap=False, displacement_k_minus_n=(1, -1)))
    g.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
    g.place_operation(ControlledZ((1, 2), 1, swap=True))
    g.place_operation(Measurement((1, 4), pi / 2))  # Measure mode 1
    g.place_operation(Measurement((2, 2), 0))  # Measure mode 0

    v0 = g.get_mode_measured_value(0)  # mode=0, coord=(2,2), b

    @feedforward
    def f1(x: float) -> float:
        return x + 1

    @feedforward
    def f2(x: float) -> float:
        return x * 2

    g.place_operation(
        PhaseRotation(
            (2, 3),
            f1(f2(v0)),
            swap=False,
            displacement_k_minus_1=(f2(f1(v0)), 0),
            displacement_k_minus_n=(0, 1),
        ),
    )

    client = construct_client(n_shots=1)
    result = execute(g, client)

    assert isinstance(result.client_result, MQC3ClientResult)
    assert result.client_result.compiled_graph is not None
    assert result.client_result.compiled_graph.n_local_macronodes == 101


@pytest.mark.skip(reason="This test needs production environment.")
def test_displacement_ff_in_emulator():
    g = GraphRepr(101, 2)
    g.place_operation(gops.Initialization((0, 0), 0, (0, BLANK_MODE)))
    g.place_operation(gops.Initialization((0, 1), 0, (1, BLANK_MODE)))

    g.place_operation(gops.Measurement((1, 0), 0))
    v = g.get_measured_value(1, 0, 0)

    @feedforward
    def ff(v: float) -> float:
        return v + 5.0

    g.place_operation(gops.Measurement((1, 1), 0, displacement_k_minus_1=(0, ff(v))))

    client = construct_client(n_shots=10000, backend="emulator")
    result = execute(g, client)

    assert isinstance(result.client_result, MQC3ClientResult)
    assert result.execution_time > timedelta(seconds=0)
    assert result.n_shots == client.n_shots

    assert result.client_result.wait_time
    assert result.client_result.wait_time > timedelta(seconds=0)
    assert result.client_result.compile_time
    assert result.client_result.compile_time > timedelta(seconds=0)
    assert result.client_result.execution_time
    assert result.client_result.execution_time > timedelta(seconds=0)
    assert result.client_result.total_time
    assert result.client_result.total_time > timedelta(seconds=0)

    er = result.execution_result
    assert isinstance(er, GraphResult)

    vals_0: list[float] = []
    vals_1: list[float] = []
    for mv in er.measured_vals:
        mvv = mv[1, 0]
        vals_0.append(mvv.m_b)

        mvv = mv[1, 1]
        vals_1.append(mvv.m_b)

    ave = [np.mean(vals_0), np.mean(vals_1)]
    var_0 = np.var(vals_0)
    var_1 = np.var([vals_1[i] - ff(vals_0[i]) for i in range(len(vals_0))])  # type: ignore  # noqa: PGH003

    assert abs(ff(ave[0]) - ave[1]) < 0.1  # type: ignore  # noqa: PGH003
    assert abs(var_0 - var_1) < 1


def test_100_nodes_no_wrap():
    # Construct base graph.
    g = GraphRepr(100, 1)
    g.place_operation(Initialization(macronode=(0, 0), theta=0.0, initialized_modes=(0, BLANK_MODE)))
    g.place_operation(Measurement(macronode=(99, 0), theta=0.0))

    # Compose with n_shots = 3
    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=3, n_local_macronodes=101, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True, scale=15, fontsize=3)
    assert map_info.n_shots == 3
    assert n_execs_needed == 1

    client = construct_client(n_shots=n_execs_needed, backend="emulator")
    res = execute(g_composite, client)

    assert isinstance(res.client_result, MQC3ClientResult)
    composite_result = res.client_result.graph_result
    assert isinstance(composite_result, GraphResult)
    decomposed_result = decompose_composite_graph_result(composite_result, map_info, g)
    assert len(decomposed_result) == 3
    for smv in decomposed_result:
        assert len(smv) == 1

    # Compose with n_shots = 8
    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=8, n_local_macronodes=101, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True, scale=15, fontsize=3)
    assert map_info.n_shots == 3
    assert n_execs_needed == 3

    client = construct_client(n_shots=n_execs_needed, backend="emulator")
    res = execute(g_composite, client)

    assert isinstance(res.client_result, MQC3ClientResult)
    composite_result = res.client_result.graph_result
    assert isinstance(composite_result, GraphResult)
    decomposed_result = decompose_composite_graph_result(composite_result, map_info, g)
    assert len(decomposed_result) == 9
    for smv in decomposed_result:
        assert len(smv) == 1


def test_101_nodes_no_wrap():
    # Construct base graph.
    g = GraphRepr(101, 1)
    g.place_operation(Initialization(macronode=(0, 0), theta=0.0, initialized_modes=(0, BLANK_MODE)))
    g.place_operation(Measurement(macronode=(100, 0), theta=0.0))

    # Compose with n_shots = 3
    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=3, n_local_macronodes=101, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True, scale=15, fontsize=3)
    assert map_info.n_shots == 3
    assert n_execs_needed == 1

    client = construct_client(n_shots=n_execs_needed, backend="emulator")
    res = execute(g_composite, client)

    assert isinstance(res.client_result, MQC3ClientResult)
    composite_result = res.client_result.graph_result
    assert isinstance(composite_result, GraphResult)
    decomposed_result = decompose_composite_graph_result(composite_result, map_info, g)
    assert len(decomposed_result) == 3
    for smv in decomposed_result:
        assert len(smv) == 1

    # Compose with n_shots = 8
    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=8, n_local_macronodes=101, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True, scale=15, fontsize=3)
    assert map_info.n_shots == 3
    assert n_execs_needed == 3

    client = construct_client(n_shots=n_execs_needed, backend="emulator")
    res = execute(g_composite, client)

    assert isinstance(res.client_result, MQC3ClientResult)
    composite_result = res.client_result.graph_result
    assert isinstance(composite_result, GraphResult)
    decomposed_result = decompose_composite_graph_result(composite_result, map_info, g)
    assert len(decomposed_result) == 9
    for smv in decomposed_result:
        assert len(smv) == 1
