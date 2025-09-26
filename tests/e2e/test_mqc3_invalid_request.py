"""Tests for executing programs with invalid requests on MQC3."""

# pyright: reportUnusedExpression=false

import re

import pytest

from mqc3.execute import execute
from mqc3.graph import GraphRepr
from mqc3.graph.ops import Measurement
from mqc3.machinery import MachineryRepr

from .common import circuit_only_intrinsics, construct_client

pytestmark = pytest.mark.network


def test_invalid_url():
    circuit = circuit_only_intrinsics()

    with pytest.raises(ValueError):  # noqa: PT011
        execute(
            circuit,
            construct_client(
                url="localhost:12345",
                n_shots=10,
            ),
        )


def test_invalid_token():
    circuit = circuit_only_intrinsics()

    with pytest.raises(RuntimeError):
        execute(
            circuit,
            construct_client(
                n_shots=10,
                token="INVALID_TOKEN",  # noqa: S106
            ),
        )


def test_graph_n_local_macronodes_over_101():
    graph = GraphRepr(102, 1)
    graph.place_operation(Measurement((0, 0), 0.0))

    err_msg = "Invalid program"
    with pytest.raises(RuntimeError, match=re.escape(err_msg)):
        execute(graph, construct_client(n_shots=1))


@pytest.mark.parametrize(
    argnames=("n_local_macronodes"),
    argvalues=[10, 100, 102],
)
def test_machinery_local_macronodes_not_101(n_local_macronodes: int):
    machinery = MachineryRepr(n_local_macronodes, 10, readout_macronode_indices={0, 1, 2})

    with pytest.raises(RuntimeError):
        execute(machinery, construct_client(n_shots=1))


@pytest.mark.longrun
def test_machinery_n_steps_over_10_000():
    n_steps = 10_001
    machinery = MachineryRepr(101, n_steps, readout_macronode_indices={0, 1, 2})

    with pytest.raises(RuntimeError) as e:
        execute(machinery, construct_client(n_shots=1))

    assert "Invalid program" in str(e.value)
