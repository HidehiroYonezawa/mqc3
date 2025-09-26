"""Test result objects in circuit representation."""

from pathlib import Path

import pytest

from mqc3.circuit.result import CircuitOperationMeasuredValue, CircuitResult, CircuitShotMeasuredValue
from mqc3.pb.io import ProtoFormat
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitResult as PbCircuitResult


def test_circuit_result():
    omv1 = CircuitOperationMeasuredValue(index=5, value=1.23)
    omv2 = CircuitOperationMeasuredValue(index=6, value=4.56)
    smv1 = CircuitShotMeasuredValue([omv1, omv2])
    smv2 = CircuitShotMeasuredValue([omv2, omv1])
    result = CircuitResult(shot_measured_values=[smv1, smv2])
    assert result.n_shots() == 2
    assert result.get_shot_measured_value(1)[0].value == pytest.approx(4.56)
    for r in result:
        assert isinstance(r, CircuitShotMeasuredValue)
    assert smv1.n_operations() == 2
    assert isinstance(result.proto(), PbCircuitResult)
    assert isinstance(CircuitResult.construct_from_proto(result.proto()), CircuitResult)


def test_circuit_result_error():
    omv1 = CircuitOperationMeasuredValue(index=5, value=1.23)
    omv2 = CircuitOperationMeasuredValue(index=6, value=4.56)
    smv1 = CircuitShotMeasuredValue([omv1, omv2])
    smv2 = CircuitShotMeasuredValue([omv2, omv1])
    result = CircuitResult(shot_measured_values=[smv1, smv2])
    with pytest.raises(TypeError):
        CircuitResult(shot_measured_values=[smv1, smv2, 1])  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="Index is invalid"):
        result.get_shot_measured_value(-1)
    with pytest.raises(ValueError, match="Index is invalid"):
        result.get_shot_measured_value(2)


@pytest.mark.parametrize("proto_format", ["text", "json", "binary"])
def test_save_and_load(tmp_path: Path, proto_format: ProtoFormat):
    omv1 = CircuitOperationMeasuredValue(index=5, value=1.23)
    omv2 = CircuitOperationMeasuredValue(index=6, value=4.56)
    smv1 = CircuitShotMeasuredValue([omv1, omv2])
    smv2 = CircuitShotMeasuredValue([omv2, omv1])
    result = CircuitResult(shot_measured_values=[smv1, smv2])

    temp_filepath = tmp_path / "circuit_result.txt"

    # test for save method
    result.save(temp_filepath, proto_format)
    assert temp_filepath.exists()

    # test for load method
    loaded = CircuitResult.load(temp_filepath, proto_format)

    assert loaded.n_shots() == 2
    assert loaded.get_shot_measured_value(1)[0].value == pytest.approx(4.56)
    for r in loaded:
        assert isinstance(r, CircuitShotMeasuredValue)
    assert loaded.get_shot_measured_value(1).n_operations() == 2


def test_save_raise_error(tmp_path: Path):
    temp_filepath = tmp_path / "non_existent_dir" / "circuit_result.txt"
    result = CircuitResult([])
    with pytest.raises(FileNotFoundError):
        result.save(temp_filepath)

    with pytest.raises(ValueError):  # noqa: PT011
        result.save(tmp_path / "circuit_result.txt", "unsupported")  # pyright: ignore[reportArgumentType]


def test_load_raise_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CircuitResult.load("non_existent_file.txt")

    with pytest.raises(ValueError):  # noqa: PT011
        CircuitResult.load(tmp_path / "circuit_result.txt", "unsupported")  # pyright: ignore[reportArgumentType]
