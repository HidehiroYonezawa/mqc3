"""Tests for executing circuits with MQC3Client and SimulatorClient using SSL/TLS certificates.

This module verifies that:
- Clients can successfully connect with valid certificates or system defaults.
- Connections fail properly with invalid certificates.
"""

from pathlib import Path

import pytest

from mqc3.client import SimulatorClient
from mqc3.execute import execute

from .common import circuit_only_intrinsics, construct_client, teleportation_circuit


@pytest.mark.network
def test_mqc3_client_with_default_cert():
    client = construct_client(n_shots=1, ca_cert_path=None)

    result = execute(circuit_only_intrinsics(), client)
    assert result is not None


@pytest.mark.network
def test_mqc3_client_with_valid_ca_cert_path():
    client = construct_client(
        n_shots=1, ca_cert_path=str(Path(__file__).parent.resolve() / "certs" / "AmazonRootCA1.pem")
    )

    result = execute(circuit_only_intrinsics(), client)
    assert result is not None


@pytest.mark.network
def test_mqc3_client_with_invalid_ca_cert_path():
    client = construct_client(
        n_shots=1, ca_cert_path=str(Path(__file__).parent.resolve() / "certs" / "invalid_cert.pem")
    )

    with pytest.raises(ValueError) as e:  # noqa: PT011
        execute(circuit_only_intrinsics(), client)
    assert "Cannot invoke RPC on closed channel!" in str(e.value)


@pytest.mark.simulator
def test_simulator_client_with_default_cert():
    client = SimulatorClient(n_shots=1, state_save_policy="all", ca_cert_file=None)

    result = execute(teleportation_circuit(disp=(0, 0)), client)
    assert result is not None


@pytest.mark.simulator
def test_simulator_client_with_valid_ca_cert_path():
    client = SimulatorClient(
        n_shots=1,
        state_save_policy="all",
        ca_cert_file=str(Path(__file__).parent.resolve() / "certs" / "AmazonRootCA1.pem"),
    )

    result = execute(teleportation_circuit(disp=(0, 0)), client)
    assert result is not None


@pytest.mark.simulator
def test_simulator_client_with_invalid_ca_cert_path():
    client = SimulatorClient(
        n_shots=1,
        state_save_policy="all",
        ca_cert_file=str(Path(__file__).parent.resolve() / "certs" / "invalid_cert.pem"),
    )

    with pytest.raises(ValueError) as e:  # noqa: PT011
        execute(teleportation_circuit(disp=(0, 0)), client)
    assert "Cannot invoke RPC on closed channel!" in str(e.value)
