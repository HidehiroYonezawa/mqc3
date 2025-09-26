"""Test graph representation."""

import random
from pathlib import Path

import pytest
from numpy import pi

from mqc3.feedforward import FeedForward, feedforward
from mqc3.graph import GraphRepr, Wiring
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    ArbitraryFirst,
    ArbitrarySecond,
    BeamSplitter,
    ControlledZ,
    Initialization,
    Measurement,
    PhaseRotation,
    ShearXInvariant,
)
from mqc3.pb.io import ProtoFormat
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphRepresentation as PbGraphRepr

from .common import make_sample_graph


def test_program_error():
    g = GraphRepr(1, 3)
    for i in range(1):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((0, 2), 0.0, 0.0, 0.0, swap=False))
    isinstance(g.proto(), PbGraphRepr)
    isinstance(GraphRepr.construct_from_proto(g.proto()), GraphRepr)
    with pytest.raises(IndexError, match="`h` must be in the range"):
        g.get_index(-1, 0)
    with pytest.raises(IndexError, match="`w` must be in the range"):
        g.get_index(0, -1)
    with pytest.raises(IndexError, match="`h` must be in the range"):
        g.get_index(1, 0)
    with pytest.raises(IndexError, match="`w` must be in the range"):
        g.get_index(0, 3)
    with pytest.raises(IndexError, match="`i` must be in the range"):
        g.get_coord(-1)
    with pytest.raises(IndexError, match="`i` must be in the range"):
        g.get_coord(3)
    with pytest.raises(ValueError, match="Invalid macronode position"):
        g.place_operation(PhaseRotation((1, 1), 1, swap=False))
    with pytest.raises(ValueError, match="Invalid macronode position"):
        g.place_operation(PhaseRotation((1, 3), 1, swap=False))
    with pytest.raises(ValueError, match="is greater than the current value"):
        g.reduce_steps(4)
    with pytest.raises(ValueError, match="The target mode index must not be"):
        g.calc_mode_operations(BLANK_MODE)
    with pytest.raises(ValueError, match="is not found"):
        g.calc_mode_operations(1)
    g.calc_mode_operations(0)
    assert isinstance(g.proto(), PbGraphRepr)
    isinstance(GraphRepr.construct_from_proto(g.proto()), GraphRepr)


def test_all_through():
    g = GraphRepr(5, 7)
    for i in range(3):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))

    assert g.n_local_macronodes == 5
    assert g.n_steps == 7
    assert g.n_total_macronodes == 35

    assert g.get_index(3, 0) == 3
    assert g.get_index(0, 1) == 5
    assert g.get_coord(3) == (3, 0)
    assert g.get_coord(5) == (0, 1)

    for j in range(1, 7):
        assert g.calc_io_of_macronode(0, j) == (0, BLANK_MODE, 0, BLANK_MODE)
        assert g.calc_io_of_macronode(1, j) == (1, BLANK_MODE, 1, BLANK_MODE)
        assert g.calc_io_of_macronode(2, j) == (2, BLANK_MODE, 2, BLANK_MODE)
        assert g.calc_io_of_macronode(3, j) == (BLANK_MODE, BLANK_MODE, BLANK_MODE, BLANK_MODE)
        assert g.calc_io_of_macronode(4, j) == (BLANK_MODE, BLANK_MODE, BLANK_MODE, BLANK_MODE)

    print(g.proto())


def test_wire_random():
    g = GraphRepr(6, 6)
    for i in range(4):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i), readout=True))
    for i in range(6):
        for j in range(1, 6):
            g.place_operation(Wiring((i, j), swap=random.randint(0, 1) == 0))

    assert g.n_local_macronodes == 6
    assert g.n_steps == 6
    assert g.n_total_macronodes == 36

    assert g.get_index(3, 0) == 3
    assert g.get_index(0, 1) == 6
    assert g.get_coord(3) == (3, 0)
    assert g.get_coord(6) == (0, 1)

    print(g.proto())
    reconstructed = GraphRepr.construct_from_proto(g.proto())

    assert reconstructed.n_local_macronodes == 6
    assert reconstructed.n_steps == 6
    assert reconstructed.n_total_macronodes == 36
    assert isinstance(reconstructed.get_operation(1, 1), Wiring)
    assert isinstance(reconstructed.get_operation(1, 0), Initialization)

    op_init_args = reconstructed.get_operation(1, 0)._get_init_args()  # noqa: SLF001
    assert op_init_args["macronode"] == (1, 0)
    assert op_init_args["theta"] == 0.0
    assert op_init_args["initialized_modes"] == (BLANK_MODE, 1)
    assert op_init_args["readout"] is True


def test_sample_graph():
    g = make_sample_graph()

    assert g.n_local_macronodes == 5
    assert g.n_steps == 6
    assert g.n_total_macronodes == 30

    reconstructed = GraphRepr.construct_from_proto(g.proto())
    assert isinstance(reconstructed.get_operation(0, 1), PhaseRotation)
    assert isinstance(reconstructed.get_operation(1, 1), ControlledZ)
    assert isinstance(reconstructed.get_operation(4, 1), BeamSplitter)
    assert isinstance(reconstructed.get_operation(2, 2), Measurement)
    assert isinstance(reconstructed.get_operation(3, 2), ShearXInvariant)
    assert isinstance(reconstructed.get_operation(4, 2), PhaseRotation)
    assert isinstance(reconstructed.get_operation(3, 3), BeamSplitter)
    assert isinstance(reconstructed.get_operation(1, 4), ControlledZ)
    assert isinstance(reconstructed.get_operation(2, 4), Measurement)
    assert isinstance(reconstructed.get_operation(3, 4), Measurement)
    assert isinstance(reconstructed.get_operation(4, 4), Measurement)
    assert isinstance(reconstructed.get_operation(1, 5), Measurement)


def test_arbitrary_through():
    g = GraphRepr(1, 3)
    for i in range(1):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((0, 2), 0.0, 0.0, 0.0, swap=False))

    assert g.n_local_macronodes == 1
    assert g.n_steps == 3
    assert g.n_total_macronodes == 3

    print(g.proto())


def test_arbitrary_swap():
    g = GraphRepr(1, 3)
    for i in range(1):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((0, 2), 0.0, 0.0, 0.0, swap=True))

    assert g.n_local_macronodes == 1
    assert g.n_steps == 3
    assert g.n_total_macronodes == 3

    print(g.proto())


def test_calc_io_of_macronode_arbitrary():
    g = GraphRepr(9, 9)
    g.place_operation(Initialization(macronode=(0, 0), theta=0.0, initialized_modes=(BLANK_MODE, 0)))

    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitrarySecond((1, 1), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitraryFirst((1, 2), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitrarySecond((2, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitraryFirst((3, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((4, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitraryFirst((5, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((6, 2), 0.0, 0.0, 0.0, swap=True))

    g.place_operation(ArbitraryFirst((6, 3), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((6, 4), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitraryFirst((6, 5), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((6, 6), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitraryFirst((7, 6), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitrarySecond((7, 7), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitraryFirst((8, 7), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitrarySecond((8, 8), 0.0, 0.0, 0.0, swap=False))

    assert g.calc_io_of_macronode(0, 0) == (BLANK_MODE, BLANK_MODE, 0, BLANK_MODE)

    assert g.calc_io_of_macronode(0, 1) == (0, BLANK_MODE, BLANK_MODE, 0)
    assert g.calc_io_of_macronode(1, 1) == (BLANK_MODE, 0, 0, BLANK_MODE)
    assert g.calc_io_of_macronode(1, 2) == (0, BLANK_MODE, BLANK_MODE, 0)
    assert g.calc_io_of_macronode(2, 2) == (BLANK_MODE, 0, BLANK_MODE, 0)
    assert g.calc_io_of_macronode(3, 2) == (BLANK_MODE, 0, BLANK_MODE, 0)
    assert g.calc_io_of_macronode(4, 2) == (BLANK_MODE, 0, BLANK_MODE, 0)
    assert g.calc_io_of_macronode(5, 2) == (BLANK_MODE, 0, BLANK_MODE, 0)
    assert g.calc_io_of_macronode(6, 2) == (BLANK_MODE, 0, 0, BLANK_MODE)

    assert g.calc_io_of_macronode(6, 3) == (0, BLANK_MODE, 0, BLANK_MODE)
    assert g.calc_io_of_macronode(6, 4) == (0, BLANK_MODE, 0, BLANK_MODE)
    assert g.calc_io_of_macronode(6, 5) == (0, BLANK_MODE, 0, BLANK_MODE)
    assert g.calc_io_of_macronode(6, 6) == (0, BLANK_MODE, BLANK_MODE, 0)
    assert g.calc_io_of_macronode(7, 6) == (BLANK_MODE, 0, 0, BLANK_MODE)
    assert g.calc_io_of_macronode(7, 7) == (0, BLANK_MODE, BLANK_MODE, 0)
    assert g.calc_io_of_macronode(8, 7) == (BLANK_MODE, 0, 0, BLANK_MODE)
    assert g.calc_io_of_macronode(8, 8) == (0, BLANK_MODE, 0, BLANK_MODE)


def test_calc_mode_operations_sample_graph():
    g = make_sample_graph()

    mode_operations = g.calc_mode_operations(mode=0)
    assert len(mode_operations) == 4
    assert isinstance(mode_operations[0], Initialization)
    assert isinstance(mode_operations[1], PhaseRotation)
    assert isinstance(mode_operations[2], ControlledZ)
    assert isinstance(mode_operations[3], Measurement)

    mode_operations = g.calc_mode_operations(mode=1)
    assert len(mode_operations) == 4
    assert isinstance(mode_operations[0], Initialization)
    assert isinstance(mode_operations[1], ControlledZ)
    assert isinstance(mode_operations[2], ControlledZ)
    assert isinstance(mode_operations[3], Measurement)

    mode_operations = g.calc_mode_operations(mode=2)
    assert len(mode_operations) == 4
    assert isinstance(mode_operations[0], Initialization)
    assert isinstance(mode_operations[1], BeamSplitter)
    assert isinstance(mode_operations[2], BeamSplitter)
    assert isinstance(mode_operations[3], Measurement)

    mode_operations = g.calc_mode_operations(mode=3)
    assert len(mode_operations) == 4
    assert isinstance(mode_operations[0], Initialization)
    assert isinstance(mode_operations[1], ShearXInvariant)
    assert isinstance(mode_operations[2], BeamSplitter)
    assert isinstance(mode_operations[3], Measurement)

    mode_operations = g.calc_mode_operations(mode=4)
    assert len(mode_operations) == 5
    assert isinstance(mode_operations[0], Initialization)
    assert isinstance(mode_operations[1], BeamSplitter)
    assert isinstance(mode_operations[2], PhaseRotation)
    assert isinstance(mode_operations[3], ControlledZ)
    assert isinstance(mode_operations[4], Measurement)


def test_calc_mode_operations_arbitrary():
    g = GraphRepr(9, 9)
    g.place_operation(Initialization(macronode=(0, 0), theta=0.0, initialized_modes=(BLANK_MODE, 0)))

    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitrarySecond((1, 1), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitraryFirst((1, 2), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitrarySecond((2, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitraryFirst((3, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((4, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitraryFirst((5, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((6, 2), 0.0, 0.0, 0.0, swap=True))

    g.place_operation(ArbitraryFirst((6, 3), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((6, 4), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitraryFirst((6, 5), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((6, 6), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitraryFirst((7, 6), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitrarySecond((7, 7), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitraryFirst((8, 7), 0.0, 0.0, 0.0, swap=True))
    g.place_operation(ArbitrarySecond((8, 8), 0.0, 0.0, 0.0, swap=False))

    mode_operations = g.calc_mode_operations(mode=0)
    assert len(mode_operations) == 17

    assert isinstance(mode_operations[0], Initialization)

    swap_list = [
        (True, True),
        (True, False),
        (False, False),
        (False, True),
        (False, False),
        (False, True),
        (True, True),
    ]

    for i, (swap1, swap2) in enumerate(swap_list):
        assert isinstance(mode_operations[i * 2 + 1], ArbitraryFirst)
        assert mode_operations[i * 2 + 1].swap == swap1

        assert isinstance(mode_operations[i * 2 + 2], ArbitrarySecond)
        assert mode_operations[i * 2 + 2].swap == swap2


def test_increase_local_macronodes_sample_graph():
    g = make_sample_graph()

    mode_operations = [g.calc_mode_operations(mode=i) for i in range(5)]

    g.increase_local_macronodes(5)
    assert g.n_local_macronodes == 5
    assert g.n_steps == 6
    assert g.n_total_macronodes == 30
    assert mode_operations == [g.calc_mode_operations(mode=i) for i in range(5)]

    g.increase_local_macronodes(6)
    assert g.n_local_macronodes == 6
    assert g.n_steps == 6
    assert g.n_total_macronodes == 36
    assert mode_operations == [g.calc_mode_operations(mode=i) for i in range(5)]

    g.increase_local_macronodes(101)
    assert g.n_local_macronodes == 101
    assert g.n_steps == 6
    assert g.n_total_macronodes == 606
    assert mode_operations == [g.calc_mode_operations(mode=i) for i in range(5)]


def test_increase_local_macronodes_arbitrary():
    g = GraphRepr(4, 5)
    for i in range(2):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((0, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitraryFirst((0, 3), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((0, 4), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitraryFirst((1, 1), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((1, 2), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitraryFirst((1, 3), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((1, 4), 0.0, 0.0, 0.0, swap=True))

    mode_operations = [g.calc_mode_operations(mode=i) for i in range(2)]

    g.increase_local_macronodes(5)
    assert g.n_local_macronodes == 5
    assert g.n_steps == 5
    assert g.n_total_macronodes == 25
    assert mode_operations == [g.calc_mode_operations(mode=i) for i in range(2)]

    g.increase_local_macronodes(7)
    assert g.n_local_macronodes == 7
    assert g.n_steps == 5
    assert g.n_total_macronodes == 35
    assert mode_operations == [g.calc_mode_operations(mode=i) for i in range(2)]

    g.increase_local_macronodes(101)
    assert g.n_local_macronodes == 101
    assert g.n_steps == 5
    assert g.n_total_macronodes == 505
    assert mode_operations == [g.calc_mode_operations(mode=i) for i in range(2)]


def test_increase_local_macronodes_exception():
    g = GraphRepr(n_local_macronodes=10, n_steps=5)
    with pytest.raises(
        expected_exception=ValueError,
        match=r"`new_n_local_macronodes` \(9\) is smaller than the current value \(10\)\.",
    ):
        g.increase_local_macronodes(new_n_local_macronodes=9)


@pytest.fixture
def test_graph():
    g = GraphRepr(3, 4)

    g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(PhaseRotation((1, 1), pi / 2, swap=False, displacement_k_minus_n=(1, -1)))
    g.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
    g.place_operation(ControlledZ((1, 2), 1, swap=True))
    g.place_operation(Measurement((1, 3), pi / 2))
    g.place_operation(Measurement((2, 2), 0))

    return g


@pytest.mark.parametrize("proto_format", ["text", "json", "binary"])
def test_save_and_load(test_graph: GraphRepr, tmp_path: Path, proto_format: ProtoFormat):
    temp_filepath = tmp_path / "graph_data.txt"

    # test for save method
    test_graph.save(temp_filepath, proto_format)
    assert temp_filepath.exists()

    # test for load method
    g = GraphRepr.load(temp_filepath, proto_format)

    assert g.n_local_macronodes == 3
    assert g.n_steps == 4
    assert g.n_total_macronodes == 12

    assert g.get_index(2, 0) == 2
    assert g.get_index(0, 1) == 3
    assert g.get_coord(2) == (2, 0)
    assert g.get_coord(3) == (0, 1)


def test_save_raise_error(test_graph: GraphRepr, tmp_path: Path):
    temp_filepath = tmp_path / "non_existent_dir" / "graph_data.txt"
    with pytest.raises(FileNotFoundError):
        test_graph.save(temp_filepath)

    with pytest.raises(ValueError):  # noqa: PT011
        test_graph.save(tmp_path / "graph_data.txt", "unsupported")  # pyright: ignore[reportArgumentType]


def test_load_raise_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        GraphRepr.load("non_existent_file.txt")

    with pytest.raises(ValueError):  # noqa: PT011
        GraphRepr.load(tmp_path / "graph_data.txt", "unsupported")  # pyright: ignore[reportArgumentType]


def test_feedforward():
    g = GraphRepr(3, 4)

    g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(PhaseRotation((1, 1), pi / 2, swap=False, displacement_k_minus_n=(1, BLANK_MODE)))
    g.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
    g.place_operation(ControlledZ((1, 2), 1, swap=True))
    g.place_operation(Measurement((1, 3), pi / 2))
    g.place_operation(Measurement((2, 2), 0))

    v = g.get_measured_value(1, 3, 1)

    @feedforward
    def f1(x: float) -> float:
        return x + 1

    @feedforward
    def f2(x: float) -> float:
        return x * 2

    g.place_operation(
        PhaseRotation(
            (2, 3),
            f1(v),
            swap=False,
            displacement_k_minus_1=(f2(v), 0),
            displacement_k_minus_n=(0, 1),
        ),
    )
    op = g.get_operation(2, 3)
    assert isinstance(op, PhaseRotation)
    assert isinstance(op.parameters[0], FeedForward)
    assert isinstance(op.displacement_k_minus_1[0], FeedForward)
    assert isinstance(op.displacement_k_minus_1[1], int)
    assert isinstance(op.displacement_k_minus_n[0], int)
    assert isinstance(op.displacement_k_minus_n[1], int)

    assert op.parameters[0].func(3) == 4
    assert op.displacement_k_minus_1[0].func(3) == 6


def test_feedforward_proto():
    g = GraphRepr(3, 4)

    g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(PhaseRotation((1, 1), pi / 2, swap=False, displacement_k_minus_n=(1, -1)))
    g.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
    g.place_operation(ControlledZ((1, 2), 1, swap=True))
    g.place_operation(Measurement((1, 3), pi / 2))

    v = g.get_measured_value(1, 3, 1)

    @feedforward
    def f1(x: float) -> float:
        return x + 1

    @feedforward
    def f2(x: float) -> float:
        return x * 2

    g.place_operation(
        PhaseRotation(
            (2, 3),
            f1(v),
            swap=False,
            displacement_k_minus_1=(f2(v), 0),
            displacement_k_minus_n=(0, 1),
        ),
    )

    v = g.get_mode_measured_value(0)
    g.place_operation(Measurement((2, 2), f2(v)))

    pb_graph = g.proto()
    reconstructed = GraphRepr.construct_from_proto(pb_graph)

    op = reconstructed.get_operation(2, 3)
    assert isinstance(op, PhaseRotation)
    assert isinstance(op.parameters[0], FeedForward)
    assert isinstance(op.displacement_k_minus_1[0], FeedForward)
    assert isinstance(op.displacement_k_minus_1[1], float)
    assert isinstance(op.displacement_k_minus_n[0], float)
    assert isinstance(op.displacement_k_minus_n[1], float)

    assert op.parameters[0].func(3) == 4
    assert op.displacement_k_minus_1[0].func(3) == 6
    assert op.parameters[0].variable.get_from_operation() == (1, 3, 1)

    op = reconstructed.get_operation(2, 2)
    assert isinstance(op, Measurement)
    assert isinstance(op.parameters[0], FeedForward)
    assert op.parameters[0].func(3) == 6
