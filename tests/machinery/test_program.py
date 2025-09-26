"""Test machinery representation."""

import random
from itertools import product
from math import pi
from pathlib import Path

import numpy as np
import pytest

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
from mqc3.machinery.macronode_angle import MacronodeAngle
from mqc3.machinery.program import FFCoeffMatrixGenerationMethods, MachineryRepr
from mqc3.pb.io import ProtoFormat
from mqc3.pb.mqc3_cloud.program.v1.machinery_pb2 import MachineryRepresentation as PbMachineryRepr


def test_all_through():  # noqa: C901
    g = GraphRepr(5, 7)
    for i in range(3):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    for i in range(3):
        g.place_operation(Measurement(macronode=(i, 6), theta=0.0))

    machinery_repr = MachineryRepr.from_graph_repr(g)

    assert machinery_repr.n_local_macronodes == 5
    assert machinery_repr.n_steps == 7
    assert machinery_repr.n_total_macronodes == 35

    for ind in range(machinery_repr.n_total_macronodes):
        assert machinery_repr.get_homodyne_angle(ind) is not None

    for narray in machinery_repr.ff_coeff_matrix_k_plus_1:
        assert len(narray) == 4
        assert len(narray[0]) == 4
    for narray in machinery_repr.ff_coeff_matrix_k_plus_n:
        assert len(narray) == 4
        assert len(narray[0]) == 4

    assert len(machinery_repr.displacements_k_minus_1) == 35
    assert len(machinery_repr.displacements_k_minus_n) == 35
    for d in machinery_repr.displacements_k_minus_1:
        assert d == (0, 0)
    for d in machinery_repr.displacements_k_minus_n:
        assert d == (0, 0)

    assert set(machinery_repr.readout_macronode_indices) == {6 * 5 + 0, 6 * 5 + 1, 6 * 5 + 2}

    # Test macronode angles
    measurement = MacronodeAngle(0, 0, 0, 0)
    wiring = MacronodeAngle(0, pi / 2, 0, pi / 2)
    for h, w in product(range(5), range(7)):
        angle = machinery_repr.get_homodyne_angle(w * 5 + h)
        if (w in {0, 6}) and h < 3:
            for i in range(4):
                assert angle[i] == measurement[i]
        else:
            for i in range(4):
                assert angle[i] == wiring[i]

    proto = machinery_repr.proto()
    assert isinstance(proto, PbMachineryRepr)
    assert isinstance(MachineryRepr.construct_from_proto(proto), MachineryRepr)


def test_empty_wire_policy_measure():  # noqa: C901
    g = GraphRepr(5, 7)
    for i in range(3):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    for i in range(3):
        g.place_operation(Measurement(macronode=(i, 6), theta=0.0))

    machinery_repr = MachineryRepr.from_graph_repr(g, empty_wire_policy="measure")

    assert machinery_repr.n_local_macronodes == 5
    assert machinery_repr.n_steps == 7
    assert machinery_repr.n_total_macronodes == 35

    for ind in range(machinery_repr.n_total_macronodes):
        assert machinery_repr.get_homodyne_angle(ind) is not None

    for narray in machinery_repr.ff_coeff_matrix_k_plus_1:
        assert len(narray) == 4
        assert len(narray[0]) == 4
    for narray in machinery_repr.ff_coeff_matrix_k_plus_n:
        assert len(narray) == 4
        assert len(narray[0]) == 4

    assert len(machinery_repr.displacements_k_minus_1) == 35
    assert len(machinery_repr.displacements_k_minus_n) == 35
    for d in machinery_repr.displacements_k_minus_1:
        assert d == (0, 0)
    for d in machinery_repr.displacements_k_minus_n:
        assert d == (0, 0)

    assert set(machinery_repr.readout_macronode_indices) == {6 * 5 + 0, 6 * 5 + 1, 6 * 5 + 2}

    # Test macronode angles
    measurement = MacronodeAngle(0, 0, 0, 0)
    wiring = MacronodeAngle(0, pi / 2, 0, pi / 2)
    for h, w in product(range(5), range(7)):
        angle = machinery_repr.get_homodyne_angle(w * 5 + h)
        if 0 < w < 6 and h < 3:
            for i in range(4):
                assert angle[i] == wiring[i]
        else:
            for i in range(4):
                assert angle[i] == measurement[i]

    proto = machinery_repr.proto()
    assert isinstance(proto, PbMachineryRepr)
    assert isinstance(MachineryRepr.construct_from_proto(proto), MachineryRepr)


def test_wire_random():
    g = GraphRepr(6, 6)
    for i in range(4):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    for i in range(6):
        for j in range(1, 6):
            g.place_operation(Wiring((i, j), swap=random.randint(0, 1) == 0))

    machinery_repr = MachineryRepr.from_graph_repr(g)

    assert machinery_repr.n_local_macronodes == 6
    assert machinery_repr.n_steps == 6
    assert machinery_repr.n_total_macronodes == 36

    for ind in range(machinery_repr.n_total_macronodes):
        assert machinery_repr.get_homodyne_angle(ind) is not None

    for narray in machinery_repr.ff_coeff_matrix_k_plus_1:
        assert len(narray) == 4
        assert len(narray[0]) == 4
    for narray in machinery_repr.ff_coeff_matrix_k_plus_n:
        assert len(narray) == 4
        assert len(narray[0]) == 4

    assert len(machinery_repr.displacements_k_minus_1) == 36
    assert len(machinery_repr.displacements_k_minus_n) == 36
    for d in machinery_repr.displacements_k_minus_1:
        assert d == (0, 0)
    for d in machinery_repr.displacements_k_minus_n:
        assert d == (0, 0)

    proto = machinery_repr.proto()
    assert isinstance(proto, PbMachineryRepr)
    assert isinstance(MachineryRepr.construct_from_proto(proto), MachineryRepr)


def test_sample():
    g = GraphRepr(5, 6)
    for i in range(5):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))

    # w=1
    g.place_operation(PhaseRotation((0, 1), 0.0, swap=True))
    g.place_operation(ControlledZ((1, 1), 0.0, swap=False))
    g.place_operation(Wiring((2, 1), swap=True))
    g.place_operation(Wiring((3, 1), swap=False))
    g.place_operation(BeamSplitter((4, 1), 0.0, 0.0, swap=False))
    # w=2
    g.place_operation(Wiring((0, 2), swap=True))
    g.place_operation(Wiring((1, 2), swap=False))
    g.place_operation(Measurement((2, 2), 0.0))
    g.place_operation(ShearXInvariant((3, 2), 0.0, swap=False))
    g.place_operation(PhaseRotation((4, 2), 0.0, swap=False))
    # w=3
    g.place_operation(Wiring((0, 3), swap=True))
    g.place_operation(Wiring((1, 3), swap=False))
    g.place_operation(Wiring((2, 3), swap=False))
    g.place_operation(BeamSplitter((3, 3), 0.0, 0.0, swap=False))
    g.place_operation(Wiring((4, 3), swap=True))
    # w=4
    g.place_operation(Wiring((0, 4), swap=False))
    g.place_operation(ControlledZ((1, 4), 0.0, swap=False))
    g.place_operation(Measurement((2, 4), 0.0))
    g.place_operation(Measurement((3, 4), 0.0))
    g.place_operation(Measurement((4, 4), 0.0))
    # w=5
    g.place_operation(Wiring((0, 5), swap=False))
    g.place_operation(Measurement((1, 5), 0.0))
    g.place_operation(Wiring((2, 5), swap=False))
    g.place_operation(Wiring((3, 5), swap=False))
    g.place_operation(Wiring((4, 5), swap=False))

    machinery_repr = MachineryRepr.from_graph_repr(g)

    assert machinery_repr.n_local_macronodes == 5
    assert machinery_repr.n_steps == 6
    assert machinery_repr.n_total_macronodes == 30

    for ind in range(machinery_repr.n_total_macronodes):
        assert machinery_repr.get_homodyne_angle(ind) is not None

    for narray in machinery_repr.ff_coeff_matrix_k_plus_1:
        assert len(narray) == 4
        assert len(narray[0]) == 4
    for narray in machinery_repr.ff_coeff_matrix_k_plus_n:
        assert len(narray) == 4
        assert len(narray[0]) == 4

    assert len(machinery_repr.displacements_k_minus_1) == 30
    assert len(machinery_repr.displacements_k_minus_n) == 30
    for d in machinery_repr.displacements_k_minus_1:
        assert d == (0, 0)
    for d in machinery_repr.displacements_k_minus_n:
        assert d == (0, 0)

    proto = machinery_repr.proto()
    assert isinstance(proto, PbMachineryRepr)
    assert isinstance(MachineryRepr.construct_from_proto(proto), MachineryRepr)


def test_arbitrary_through():
    g = GraphRepr(1, 3)
    for i in range(1):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((0, 2), 0.0, 0.0, 0.0, swap=False))

    machinery_repr = MachineryRepr.from_graph_repr(g)

    assert machinery_repr.n_local_macronodes == 1
    assert machinery_repr.n_steps == 3
    assert machinery_repr.n_total_macronodes == 3

    for ind in range(machinery_repr.n_total_macronodes):
        assert machinery_repr.get_homodyne_angle(ind) is not None

    for narray in machinery_repr.ff_coeff_matrix_k_plus_1:
        assert len(narray) == 4
        assert len(narray[0]) == 4
    for narray in machinery_repr.ff_coeff_matrix_k_plus_n:
        assert len(narray) == 4
        assert len(narray[0]) == 4

    assert len(machinery_repr.displacements_k_minus_1) == 3
    assert len(machinery_repr.displacements_k_minus_n) == 3
    for d in machinery_repr.displacements_k_minus_1:
        assert d == (0, 0)
    for d in machinery_repr.displacements_k_minus_n:
        assert d == (0, 0)

    proto = machinery_repr.proto()
    assert isinstance(proto, PbMachineryRepr)
    assert isinstance(MachineryRepr.construct_from_proto(proto), MachineryRepr)


def test_arbitrary_swap():
    g = GraphRepr(1, 3)
    for i in range(1):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((0, 2), 0.0, 0.0, 0.0, swap=True))

    machinery_repr = MachineryRepr.from_graph_repr(g)

    assert machinery_repr.n_local_macronodes == 1
    assert machinery_repr.n_steps == 3
    assert machinery_repr.n_total_macronodes == 3

    for ind in range(machinery_repr.n_total_macronodes):
        assert machinery_repr.get_homodyne_angle(ind) is not None

    for narray in machinery_repr.ff_coeff_matrix_k_plus_1:
        assert len(narray) == 4
        assert len(narray[0]) == 4
    for narray in machinery_repr.ff_coeff_matrix_k_plus_n:
        assert len(narray) == 4
        assert len(narray[0]) == 4

    assert len(machinery_repr.displacements_k_minus_1) == 3
    assert len(machinery_repr.displacements_k_minus_n) == 3
    for d in machinery_repr.displacements_k_minus_1:
        assert d == (0, 0)
    for d in machinery_repr.displacements_k_minus_n:
        assert d == (0, 0)

    proto = machinery_repr.proto()
    assert isinstance(proto, PbMachineryRepr)
    assert isinstance(MachineryRepr.construct_from_proto(proto), MachineryRepr)


def test_ff_matrix():
    machinery_repr = MachineryRepr(
        n_local_macronodes=2,
        n_steps=3,
        readout_macronode_indices={1, 2, 3},
    )
    assert machinery_repr.ff_coeff_matrix_k_plus_1
    assert machinery_repr.ff_coeff_matrix_k_plus_n
    machinery_repr.set_homodyne_angle(macronode_index=2, angle=MacronodeAngle(1, 2, 3, 4))
    assert machinery_repr.ff_coeff_matrix_k_plus_1
    assert machinery_repr.ff_coeff_matrix_k_plus_n


def test_do_ff():
    machinery_repr = MachineryRepr(
        n_local_macronodes=1,
        n_steps=1,
        readout_macronode_indices={0},
    )
    machinery_repr.set_homodyne_angle(
        macronode_index=0,
        angle=MacronodeAngle(pi / 2, pi / 2, pi / 2, pi / 2),
    )
    assert any(np.any(np.array(arr) != 0) for arr in machinery_repr.ff_coeff_matrix_k_plus_1)
    assert any(np.any(np.array(arr) != 0) for arr in machinery_repr.ff_coeff_matrix_k_plus_n)

    # set zero filled
    machinery_repr.ff_coeff_matrix_k_plus_1.generation_methods[0] = FFCoeffMatrixGenerationMethods.ZERO_FILLED
    machinery_repr.ff_coeff_matrix_k_plus_n.generation_methods[0] = FFCoeffMatrixGenerationMethods.ZERO_FILLED
    assert all(np.all(np.array(arr) == 0) for arr in machinery_repr.ff_coeff_matrix_k_plus_1)
    assert all(np.all(np.array(arr) == 0) for arr in machinery_repr.ff_coeff_matrix_k_plus_n)

    # measurement operation in graph repr should make the ff matrices zero filled
    n_local_macronodes = 101
    n_steps = 2
    g = GraphRepr(n_local_macronodes=n_local_macronodes, n_steps=n_steps)
    for k in range(n_local_macronodes * n_steps):
        g.place_operation(Measurement(macronode=g.get_coord(k), theta=pi / 2))
    machinery_repr = MachineryRepr.from_graph_repr(g)
    for k in range(n_local_macronodes * n_steps):
        angle = machinery_repr.get_homodyne_angle(k)
        assert angle is not None
    assert all(np.all(np.array(arr) == 0) for arr in machinery_repr.ff_coeff_matrix_k_plus_1)
    assert all(np.all(np.array(arr) == 0) for arr in machinery_repr.ff_coeff_matrix_k_plus_n)


def test_nlff():
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

    machinery_repr = MachineryRepr.from_graph_repr(g)

    m = machinery_repr.get_homodyne_angle(11)
    assert m is not None
    assert isinstance(m.theta_a, FeedForward)
    assert m.theta_a.func(2) == 1.5

    proto = machinery_repr.proto()
    assert isinstance(proto, PbMachineryRepr)
    reconstructed = MachineryRepr.construct_from_proto(proto)
    assert isinstance(reconstructed, MachineryRepr)
    m = reconstructed.get_homodyne_angle(11)
    assert m is not None
    assert isinstance(m.theta_a, FeedForward)
    assert m.theta_a.func(2) == 1.5


@pytest.fixture
def test_machinery():
    g = GraphRepr(3, 4)

    g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(PhaseRotation((1, 1), pi / 2, swap=False, displacement_k_minus_n=(1, -1)))
    g.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
    g.place_operation(ControlledZ((1, 2), 1, swap=True))
    g.place_operation(Measurement((1, 3), pi / 2))
    g.place_operation(Measurement((2, 2), 0))

    return MachineryRepr.from_graph_repr(g)


@pytest.mark.parametrize("proto_format", ["text", "json", "binary"])
def test_save_and_load(test_machinery: MachineryRepr, tmp_path: Path, proto_format: ProtoFormat):
    temp_filepath = tmp_path / "machinery_data.txt"

    # test for save method
    test_machinery.save(temp_filepath, proto_format)
    assert temp_filepath.exists()

    # test for load method
    m = MachineryRepr.load(temp_filepath, proto_format)

    assert m.n_local_macronodes == 3
    assert m.n_steps == 4
    assert m.n_total_macronodes == 12

    assert m.get_coord(2) == (2, 0)
    assert m.get_coord(3) == (0, 1)


def test_save_raise_error(test_machinery: MachineryRepr, tmp_path: Path):
    temp_filepath = tmp_path / "non_existent_dir" / "machinery_data.txt"
    with pytest.raises(FileNotFoundError):
        test_machinery.save(temp_filepath)

    with pytest.raises(ValueError):  # noqa: PT011
        test_machinery.save(tmp_path / ",machinery_data.txt", "unsupported")  # pyright: ignore[reportArgumentType]


def test_load_raise_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MachineryRepr.load("non_existent_file.txt")

    with pytest.raises(ValueError):  # noqa: PT011
        MachineryRepr.load(tmp_path / "machinery_data.txt", "unsupported")  # pyright: ignore[reportArgumentType]
