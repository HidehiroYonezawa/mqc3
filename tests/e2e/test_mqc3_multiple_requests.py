"""Tests for executing multiple circuit execution requests in parallel on MQC3."""

import multiprocessing

import pytest

from mqc3.execute import execute

from .common import circuit_only_intrinsics, construct_client

pytestmark = pytest.mark.network


def execute_circuit(index: int):
    n_shots = 100
    client = construct_client(n_shots=n_shots)
    for trial in range(3):
        circuit = circuit_only_intrinsics()

        result = execute(circuit, client)
        assert result.n_shots == n_shots
        print(f"Process: {index}, Trial: {trial} success")


def test_multiple_requests():
    process_list = [multiprocessing.Process(target=execute_circuit, args=(index,)) for index in range(5)]
    for p in process_list:
        p.start()
    print("Processes started")
    for p in process_list:
        p.join()
    print("Processes joined")
