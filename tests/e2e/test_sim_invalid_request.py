"""Tests for handling invalid inputs in the simulator client."""

import pytest

from mqc3.circuit import BosonicState, CircuitRepr
from mqc3.circuit.ops import intrinsic
from mqc3.client import SimulatorClient
from mqc3.execute import execute

pytestmark = pytest.mark.simulator


def test_invalid_shot_size():
    circuit = CircuitRepr("invalid_shot_size")
    circuit.Q(0) | intrinsic.Measurement(0)  # type: ignore  # noqa: PGH003
    for i in range(1):
        circuit.set_initial_state(i, BosonicState.vacuum())

    client = SimulatorClient(n_shots=1_000_001, state_save_policy="all")
    with pytest.raises(RuntimeError):
        execute(circuit, client)


def test_invalid_peak_prod():
    circuit = CircuitRepr("invalid_peak_prod")
    circuit.Q(20) | intrinsic.Measurement(0)  # type: ignore  # noqa: PGH003
    for i in range(21):
        circuit.set_initial_state(i, BosonicState.cat(1, 1, 0))

    client = SimulatorClient(n_shots=1, state_save_policy="none")
    with pytest.raises(RuntimeError):
        execute(circuit, client)
