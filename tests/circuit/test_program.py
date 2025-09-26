"""Test circuit representation."""

import math
from pathlib import Path

import pytest

from mqc3.circuit.ops import intrinsic, std
from mqc3.circuit.ops.intrinsic import Measurement, PhaseRotation
from mqc3.circuit.program import CircuitRepr
from mqc3.circuit.state import BosonicState
from mqc3.feedforward import feedforward
from mqc3.pb.io import ProtoFormat


def test_convert_circuit_repr():
    num_modes = 4
    c = CircuitRepr("test")
    for i in range(num_modes):
        c.Q(i) | PhaseRotation(1.0)  # pyright: ignore[reportUnusedExpression]
        c.Q(i) | Measurement(math.pi / 2)  # pyright: ignore[reportUnusedExpression]

    c.set_initial_state(0, BosonicState.vacuum())

    pb_circuit = c.proto()
    reconstructed_circuit = CircuitRepr.construct_from_proto(pb_circuit)

    assert isinstance(reconstructed_circuit, CircuitRepr)
    assert reconstructed_circuit.n_modes == num_modes

    # Check operations in the circuit representation.
    assert c.n_operations == reconstructed_circuit.n_operations
    for i in range(c.n_operations):
        assert type(c.get_operation(i)) is type(reconstructed_circuit.get_operation(i))
        c_params = c.get_operation(i).parameters()
        reconstructed_params = reconstructed_circuit.get_operation(i).parameters()
        for c_param, reconstructed_param in zip(c_params, reconstructed_params, strict=False):
            assert math.isclose(c_param, reconstructed_param, rel_tol=1e-7)  # type: ignore  # noqa: PGH003

    # Check initial states.
    assert len(c._initial_states) == len(reconstructed_circuit._initial_states)  # noqa: SLF001
    c_state = c.get_initial_state(0)
    reconstructed_circuit_state = reconstructed_circuit.get_initial_state(0)
    assert isinstance(c_state, BosonicState)
    assert c_state.is_vacuum()
    assert isinstance(reconstructed_circuit_state, BosonicState)
    assert reconstructed_circuit_state.is_vacuum()


def test_convert_circuit_repr_with_nlff_to_std_operation():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    c = CircuitRepr("test")
    m0 = c.Q(0) | Measurement(math.pi / 2)  # pyright: ignore[reportUnusedExpression]
    c.Q(1) | std.Squeezing(f(m0))  # pyright: ignore[reportUnusedExpression]

    pb_circuit = c.proto()
    assert pb_circuit.nlffs[0].from_operation == 0


def test_convert_circuit_repr_with_chained_nlff():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    @feedforward
    def square(x: float) -> float:
        return x**2

    c = CircuitRepr("test")
    m0 = c.Q(0) | Measurement(math.pi / 2)  # pyright: ignore[reportUnusedExpression]
    c.Q(1) | intrinsic.PhaseRotation(square(f(m0)))  # pyright: ignore[reportUnusedExpression]

    pb_circuit = c.proto()
    assert pb_circuit.nlffs[0].from_operation == 0


def test_convert_circuit_measurement_position_changed():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    # This circuit contains a std operation before Measurement.
    # The index of the Measurement will be changed by converting the std operation to some intrinsic operations.
    c = CircuitRepr("test")
    # std.BeamSplitter is converted to five intrinsic operations.
    c.Q(0, 1) | std.BeamSplitter(0.1, 0.2)  # pyright: ignore[reportUnusedExpression]
    m0 = c.Q(0) | Measurement(math.pi / 2)  # pyright: ignore[reportUnusedExpression]
    c.Q(2) | intrinsic.Squeezing(f(m0))  # pyright: ignore[reportUnusedExpression]

    # Check that from_operation of the nlff is expected.
    pb_circuit = c.proto()
    assert pb_circuit.nlffs[0].from_operation == 5


@pytest.fixture
def test_circuit():
    circuit = CircuitRepr("")

    circuit.Q(0) | PhaseRotation(math.pi / 2)  # pyright: ignore[reportUnusedExpression]
    circuit.Q(0, 1) | intrinsic.ControlledZ(1)  # pyright: ignore[reportUnusedExpression]
    circuit.Q(0) | Measurement(math.pi / 2)  # pyright: ignore[reportUnusedExpression]
    circuit.Q(1) | Measurement(0)  # pyright: ignore[reportUnusedExpression]

    return circuit


@pytest.mark.parametrize("proto_format", ["text", "json", "binary"])
def test_save_and_load(test_circuit: CircuitRepr, tmp_path: Path, proto_format: ProtoFormat):
    temp_filepath = tmp_path / "circuit_data.txt"

    # test for save method
    test_circuit.save(temp_filepath, proto_format)
    assert temp_filepath.exists()

    # test for load method
    c = CircuitRepr.load(temp_filepath, proto_format)

    assert c.n_modes == 2
    assert c.n_operations == 4

    assert isinstance(c.get_operation(0), PhaseRotation)
    assert isinstance(c.get_operation(1), intrinsic.ControlledZ)
    assert isinstance(c.get_operation(2), Measurement)
    assert isinstance(c.get_operation(3), Measurement)


def test_save_raise_error(test_circuit: CircuitRepr, tmp_path: Path):
    temp_filepath = tmp_path / "non_existent_dir" / "circuit_data.txt"
    with pytest.raises(FileNotFoundError):
        test_circuit.save(temp_filepath)

    with pytest.raises(ValueError):  # noqa: PT011
        test_circuit.save(tmp_path / "circuit_data.txt", "unsupported")  # pyright: ignore[reportArgumentType]


def test_load_raise_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CircuitRepr.load("non_existent_file.txt")

    with pytest.raises(ValueError):  # noqa: PT011
        CircuitRepr.load(tmp_path / "circuit_data.txt", "unsupported")  # pyright: ignore[reportArgumentType]
