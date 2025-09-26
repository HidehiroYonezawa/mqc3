"""Tests for executing circuit representations using an MQC3 client."""

from datetime import timedelta

import pytest

from mqc3.circuit import CircuitResult
from mqc3.client import MQC3ClientResult
from mqc3.execute import execute

from .common import circuit_only_intrinsics, circuit_with_feedforward, circuit_with_std, construct_client

pytestmark = pytest.mark.network


def test_execute_circuit_only_intrinsics():
    client = construct_client()
    circuit = circuit_only_intrinsics()
    result = execute(circuit, client)
    assert isinstance(result.execution_result, CircuitResult)
    assert isinstance(result.client_result, MQC3ClientResult)
    assert isinstance(result.client_result.circuit_result, CircuitResult)
    assert result.client_result.compiled_graph
    assert result.client_result.compiled_machinery
    assert result.client_result.graph_result
    assert result.client_result.machinery_result

    assert result.client_result.execution_details.scheduler_version
    assert result.client_result.execution_details.physical_lab_version
    assert result.client_result.wait_time
    assert result.client_result.wait_time > timedelta(seconds=0)
    assert result.client_result.compile_time
    assert result.client_result.compile_time > timedelta(seconds=0)
    assert result.client_result.execution_time
    assert result.client_result.execution_time > timedelta(seconds=0)
    assert result.client_result.total_time
    assert result.client_result.total_time > timedelta(seconds=0)


def test_execute_circuit_with_std():
    client = construct_client()
    circuit = circuit_with_std()
    result = execute(circuit, client)
    assert isinstance(result.execution_result, CircuitResult)
    assert isinstance(result.client_result, MQC3ClientResult)
    assert isinstance(result.client_result.circuit_result, CircuitResult)
    assert result.client_result.compiled_graph
    assert result.client_result.compiled_machinery
    assert result.client_result.graph_result
    assert result.client_result.machinery_result

    assert result.client_result.execution_details.scheduler_version
    assert result.client_result.execution_details.physical_lab_version
    assert result.client_result.wait_time
    assert result.client_result.wait_time > timedelta(seconds=0)
    assert result.client_result.compile_time
    assert result.client_result.compile_time > timedelta(seconds=0)
    assert result.client_result.execution_time
    assert result.client_result.execution_time > timedelta(seconds=0)
    assert result.client_result.total_time
    assert result.client_result.total_time > timedelta(seconds=0)


def test_execute_circuit_with_feedforward():
    client = construct_client()
    circuit = circuit_with_feedforward()
    result = execute(circuit, client)
    assert isinstance(result.execution_result, CircuitResult)
    assert isinstance(result.client_result, MQC3ClientResult)
    assert isinstance(result.client_result.circuit_result, CircuitResult)
    assert result.client_result.compiled_graph
    assert result.client_result.compiled_machinery
    assert result.client_result.graph_result
    assert result.client_result.machinery_result

    assert result.client_result.execution_details.scheduler_version
    assert result.client_result.execution_details.physical_lab_version
    assert result.client_result.wait_time
    assert result.client_result.wait_time > timedelta(seconds=0)
    assert result.client_result.compile_time
    assert result.client_result.compile_time > timedelta(seconds=0)
    assert result.client_result.execution_time
    assert result.client_result.execution_time > timedelta(seconds=0)
    assert result.client_result.total_time
    assert result.client_result.total_time > timedelta(seconds=0)
