"""Test result objects in circuit representation."""

from pathlib import Path

import pytest

from mqc3.machinery.result import (
    MachineryMacronodeMeasuredValue,
    MachineryResult,
    MachineryShotMeasuredValue,
)
from mqc3.pb.io import ProtoFormat
from mqc3.pb.mqc3_cloud.program.v1.machinery_pb2 import (
    MachineryResult as PbMachineryResult,
)


def test_machinery_result():
    mmv1 = MachineryMacronodeMeasuredValue(m_a=1.1, m_b=2.2, m_c=3.3, m_d=4.4, index=1)
    mmv2 = MachineryMacronodeMeasuredValue(5.5, 6.6, 7.7, 8.8, index=2)
    assert mmv1[1] == pytest.approx(2.2)
    assert mmv1.m_d == pytest.approx(4.4)
    assert mmv2[2] == pytest.approx(7.7)
    assert mmv2.m_a == pytest.approx(5.5)
    assert mmv1.index == 1
    assert mmv2.index == 2

    smv1 = MachineryShotMeasuredValue([mmv1, mmv2])
    smv2 = MachineryShotMeasuredValue([mmv2, mmv1])
    result = MachineryResult(shot_measured_values=[smv1, smv2])
    assert result.n_shots() == 2
    assert result.get_shot_measured_value(1)[2].m_c == pytest.approx(7.7)
    for r in result:
        assert isinstance(r, MachineryShotMeasuredValue)
        smv2_mmv1, smv2_mmv2 = tuple(_ for _ in smv2)
    assert smv2_mmv1.m_d == pytest.approx(4.4)
    assert smv2_mmv2.m_d == pytest.approx(8.8)

    with pytest.raises(TypeError):
        MachineryResult(shot_measured_values=[smv1, smv2, 1])  # pyright: ignore[reportArgumentType]


def test_machinery_result_error():
    with pytest.raises(TypeError, match="All elements must be float"):
        MachineryMacronodeMeasuredValue("a", 2, 3, 4, index=0)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="All elements must be float"):
        MachineryMacronodeMeasuredValue(1, 2, 3, "a", index=0)  # pyright: ignore[reportArgumentType]
    result = MachineryResult(
        [
            MachineryShotMeasuredValue([
                MachineryMacronodeMeasuredValue(1, 2, 3, 4, index=0),
                MachineryMacronodeMeasuredValue(5, 6, 7, 8, index=1),
            ]),
            MachineryShotMeasuredValue([
                MachineryMacronodeMeasuredValue(8, 7, 6, 5, index=0),
                MachineryMacronodeMeasuredValue(4, 3, 2, 1, index=1),
            ]),
        ],
    )
    with pytest.raises(ValueError, match="Index must be in the range"):
        result.get_shot_measured_value(-1)
    with pytest.raises(ValueError, match="Index must be in the range"):
        result.get_shot_measured_value(2)
    assert isinstance(result.proto(), PbMachineryResult)


@pytest.mark.parametrize("proto_format", ["text", "json", "binary"])
def test_save_and_load(tmp_path: Path, proto_format: ProtoFormat):
    mmv1 = MachineryMacronodeMeasuredValue(m_a=1.1, m_b=2.2, m_c=3.3, m_d=4.4, index=1)
    mmv2 = MachineryMacronodeMeasuredValue(5.5, 6.6, 7.7, 8.8, index=2)
    smv1 = MachineryShotMeasuredValue([mmv1, mmv2])
    smv2 = MachineryShotMeasuredValue([mmv2, mmv1])
    result = MachineryResult(shot_measured_values=[smv1, smv2])

    temp_filepath = tmp_path / "machinery_result.txt"
    # test for save method
    result.save(temp_filepath, proto_format)
    assert temp_filepath.exists()

    # test for load method
    loaded = MachineryResult.load(temp_filepath, proto_format)
    assert loaded.n_shots() == 2
    assert loaded.get_shot_measured_value(1)[2].m_c == pytest.approx(7.7)
    for r in loaded:
        assert isinstance(r, MachineryShotMeasuredValue)
        smv2_mmv1, smv2_mmv2 = tuple(_ for _ in smv2)
    assert smv2_mmv1.m_d == pytest.approx(4.4)
    assert smv2_mmv2.m_d == pytest.approx(8.8)


def test_save_raise_error(tmp_path: Path):
    temp_filepath = tmp_path / "non_existent_dir" / "machinery_result.txt"
    result = MachineryResult(shot_measured_values=[])
    with pytest.raises(FileNotFoundError):
        result.save(temp_filepath)

    with pytest.raises(ValueError):  # noqa: PT011
        result.save(tmp_path / "machinery_result.txt", "unsupported")  # pyright: ignore[reportArgumentType]


def test_load_raise_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MachineryResult.load("non_existent_file.txt")

    with pytest.raises(ValueError):  # noqa: PT011
        MachineryResult.load(tmp_path / "machinery_result.txt", "unsupported")  # pyright: ignore[reportArgumentType]
