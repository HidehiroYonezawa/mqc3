"""Test result objects in circuit representation."""

from pathlib import Path

import pytest

from mqc3.graph.result import GraphMacronodeMeasuredValue, GraphResult, GraphShotMeasuredValue
from mqc3.pb.io import ProtoFormat
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphResult as PbGraphResult


def test_graph_result():
    mmv1 = GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=1.2, m_d=3.4)
    mmv2 = GraphMacronodeMeasuredValue(index=2, h=0, w=1, m_b=5.6, m_d=7.8)
    smv1 = GraphShotMeasuredValue([mmv1, mmv2], n_local_macronodes=2)
    smv2 = GraphShotMeasuredValue([mmv2, mmv1], n_local_macronodes=2)
    result = GraphResult(n_local_macronodes=2, shot_measured_values=[smv1, smv2])
    assert result.n_shots() == 2
    assert result.get_shot_measured_value(0)[1].m_b == pytest.approx(1.2)
    assert result.get_shot_measured_value(1)[2].m_d == pytest.approx(7.8)
    assert result.get_shot_measured_value(0)[0, 1].m_b == pytest.approx(5.6)
    for r in result:
        assert isinstance(r, GraphShotMeasuredValue)
    assert isinstance(result.proto(), PbGraphResult)
    assert isinstance(GraphResult.construct_from_proto(result.proto()), GraphResult)
    smv2_mmv1, smv2_mmv2 = tuple(_ for _ in smv2)
    assert smv2_mmv1.m_d == pytest.approx(3.4)
    assert smv2_mmv2.m_d == pytest.approx(7.8)


def test_graph_result_error():
    mmv1 = GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=1.2, m_d=3.4)
    mmv2 = GraphMacronodeMeasuredValue(index=2, h=0, w=1, m_b=5.6, m_d=7.8)
    smv1 = GraphShotMeasuredValue([mmv1, mmv2], n_local_macronodes=2)
    smv2 = GraphShotMeasuredValue([mmv2, mmv1], n_local_macronodes=2)
    result = GraphResult(n_local_macronodes=2, shot_measured_values=[smv1, smv2])
    with pytest.raises(TypeError):
        GraphResult(n_local_macronodes=2, shot_measured_values=[smv1, smv2, 1])  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="Index is invalid"):
        result.get_shot_measured_value(-1)
    with pytest.raises(ValueError, match="Index is invalid"):
        result.get_shot_measured_value(2)


@pytest.mark.parametrize("proto_format", ["text", "json", "binary"])
def test_save_and_load(tmp_path: Path, proto_format: ProtoFormat):
    mmv1 = GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=1.2, m_d=3.4)
    mmv2 = GraphMacronodeMeasuredValue(index=2, h=0, w=1, m_b=5.6, m_d=7.8)
    smv1 = GraphShotMeasuredValue([mmv1, mmv2], n_local_macronodes=2)
    smv2 = GraphShotMeasuredValue([mmv2, mmv1], n_local_macronodes=2)
    result = GraphResult(n_local_macronodes=2, shot_measured_values=[smv1, smv2])

    temp_filepath = tmp_path / "graph_result.txt"
    # test for save method
    result.save(temp_filepath, proto_format)
    assert temp_filepath.exists()

    # test for load method
    loaded = GraphResult.load(temp_filepath, proto_format)
    assert loaded.n_shots() == 2
    assert loaded.get_shot_measured_value(0)[1].m_b == pytest.approx(1.2)
    assert loaded.get_shot_measured_value(1)[2].m_d == pytest.approx(7.8)
    assert loaded.get_shot_measured_value(0)[0, 1].m_b == pytest.approx(5.6)
    for r in loaded:
        assert isinstance(r, GraphShotMeasuredValue)


def test_save_raise_error(tmp_path: Path):
    temp_filepath = tmp_path / "non_existent_dir" / "graph_result.txt"
    result = GraphResult(n_local_macronodes=2, shot_measured_values=[])
    with pytest.raises(FileNotFoundError):
        result.save(temp_filepath)

    with pytest.raises(ValueError):  # noqa: PT011
        result.save(tmp_path / "graph_result.txt", "unsupported")  # pyright: ignore[reportArgumentType]


def test_load_raise_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        GraphResult.load("non_existent_file.txt")

    with pytest.raises(ValueError):  # noqa: PT011
        GraphResult.load(tmp_path / "graph_result.txt", "unsupported")  # pyright: ignore[reportArgumentType]
