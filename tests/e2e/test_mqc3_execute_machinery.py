"""Tests for executing machinery representations using an MQC3 client."""

# pyright: reportUnusedExpression=false

import pytest

from mqc3.client import MQC3ClientResult
from mqc3.execute import execute
from mqc3.machinery import MachineryRepr
from mqc3.machinery.result import MachineryResult

from .common import construct_client

pytestmark = pytest.mark.network


@pytest.mark.longrun
def test_machinery_mul_n_readout_and_shots_over_2_000_000():
    n_steps = 100
    machinery = MachineryRepr(101, n_steps, readout_macronode_indices=set(range(101 * n_steps)))

    result = execute(machinery, construct_client(n_shots=200))
    assert isinstance(result.execution_result, MachineryResult)
    assert isinstance(result.client_result, MQC3ClientResult)
    assert result.client_result.compiled_machinery is not None
    assert result.client_result.compiled_graph is None
    assert result.client_result.circuit_result is None
    assert result.client_result.graph_result is None
