"""Test greedy convert algorithm."""

# pyright: reportUnusedExpression=false

import matplotlib.pyplot as plt
import pytest

import mqc3.graph.ops as gops
from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic, std
from mqc3.circuit.state import HardwareConstrainedSqueezedState
from mqc3.feedforward import FeedForward, feedforward
from mqc3.graph.convert import GreedyConverter, GreedyConvertSettings
from mqc3.graph.program import GraphOpParam, ModeMeasuredVariable
from mqc3.graph.visualize import make_figure


def test_arbitrary():
    c = CircuitRepr("test_arbitrary")
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)
    c.Q(3) | intrinsic.ShearXInvariant(0.0)
    c.Q(2, 4) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(2, 3) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(4) | intrinsic.PhaseRotation(0.0)
    c.Q(1, 4) | intrinsic.ControlledZ(0.0)
    c.Q(0) | intrinsic.Arbitrary(0.0, 0.0, 0.0)
    for i in range(5):
        c.Q(i) | intrinsic.Measurement(0.0)

    for i in range(5):
        c.set_initial_state(i, HardwareConstrainedSqueezedState(i / 10))

    converter = GreedyConverter(GreedyConvertSettings(n_local_macronodes=5))
    g = converter.convert(c)
    make_figure(g)
    plt.close("all")


def test_std():
    c = CircuitRepr("test_std")
    c.Q(0) | std.Squeezing(r=0.0)
    c.Q(1, 2) | std.BeamSplitter(theta=0.0, phi=0.0)
    for i in range(3):
        c.Q(i) | intrinsic.Measurement(0.0)

    converter = GreedyConverter(GreedyConvertSettings(n_local_macronodes=5))
    g = converter.convert(c)
    make_figure(g)
    plt.close("all")


def test_displacement():
    c = CircuitRepr("test_displacement")

    # add to Initialization
    c.Q(0) | intrinsic.Displacement(1.0, 0.0)
    c.Q(0) | intrinsic.Displacement(2.0, 0.0)

    c.Q(0) | intrinsic.PhaseRotation(0.0)

    # add to ControlledZ
    c.Q(1) | intrinsic.Displacement(3.0, 0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)

    c.Q(0) | intrinsic.PhaseRotation(0.0)

    # add to PhaseRotation
    c.Q(1) | intrinsic.PhaseRotation(0.0)
    c.Q(1) | intrinsic.Displacement(0.0, 4.0)

    for i in range(2):
        c.Q(i) | intrinsic.Measurement(0.0)

    converter = GreedyConverter(GreedyConvertSettings(n_local_macronodes=2))
    g = converter.convert(c)

    # (operation type, displacement mode index, displacement_x, displacement_p)
    check_list: list[tuple[type, int, float, float]] = [
        (gops.PhaseRotation, 0, 3.0, 0.0),
        (gops.ControlledZ, 1, 3.0, 0.0),
        (gops.Measurement, 1, 0.0, 4.0),
    ]

    def is_non_zero(xp: tuple[GraphOpParam, GraphOpParam]) -> bool:
        return xp[0] != pytest.approx(0.0) or xp[1] != pytest.approx(0.0)

    for ind in range(g.n_total_macronodes):
        op = g.get_operation(*g.get_coord(ind))
        if is_non_zero(op.displacement_k_minus_1) or is_non_zero(op.displacement_k_minus_n):
            check_type, check_mode, check_x, check_p = check_list[0]

            assert isinstance(op, check_type)

            left, up, _, _ = g.calc_io_of_macronode(*g.get_coord(ind))
            if left == check_mode:
                assert op.displacement_k_minus_n[0] == pytest.approx(check_x)
                assert op.displacement_k_minus_n[1] == pytest.approx(check_p)
            elif up == check_mode:
                assert op.displacement_k_minus_1[0] == pytest.approx(check_x)
                assert op.displacement_k_minus_1[1] == pytest.approx(check_p)
            else:
                msg = "Mode mismatch"
                raise RuntimeError(msg)

            check_list.pop(0)

    if len(check_list) != 0:
        msg = "Some displacement is not set"
        raise RuntimeError(msg)


def test_feedforward():
    ff_distance_lower = 5
    ff_distance_upper = 25

    c = CircuitRepr("test_feedforward")
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)
    c.Q(3) | intrinsic.ShearXInvariant(0.0)
    c.Q(2, 4) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(2, 3) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(4) | intrinsic.PhaseRotation(0.0)
    c.Q(1, 4) | intrinsic.ControlledZ(0.0)
    x = [c.Q(i) | intrinsic.Measurement(0.0) for i in range(4)]

    @feedforward
    def f1(x: float) -> float:
        return x + 1

    @feedforward
    def f2(x: float) -> float:
        return x * 2

    c.Q(4) | intrinsic.PhaseRotation(f1(x[0]))
    c.Q(4) | intrinsic.ShearPInvariant(f2(x[1]))
    c.Q(4) | intrinsic.Measurement(f2(x[0]))

    converter = GreedyConverter(
        GreedyConvertSettings(n_local_macronodes=5, feedforward_distance=(ff_distance_lower, ff_distance_upper))
    )
    g = converter.convert(c)
    meas_0 = g.calc_mode_operations(0)[-1]
    meas_1 = g.calc_mode_operations(1)[-1]
    meas_4 = g.calc_mode_operations(4)[-1]
    assert meas_0
    assert meas_1
    assert meas_4
    assert isinstance(meas_4.parameters[0], FeedForward)
    assert meas_4.parameters[0].func(2) == 4
    ind_meas_0 = g.get_index(meas_0.macronode[0], meas_0.macronode[1])
    ind_meas_1 = g.get_index(meas_1.macronode[0], meas_1.macronode[1])
    for op in g.calc_mode_operations(4):
        h, w = op.macronode
        ind_op = g.get_index(h, w)
        for p in op.parameters:
            if not isinstance(p, FeedForward):
                continue
            assert isinstance(p.variable, ModeMeasuredVariable)
            match p.variable.mode:
                case 0:
                    assert ind_meas_0 + ff_distance_lower <= ind_op <= ind_meas_0 + ff_distance_upper
                case 1:
                    assert ind_meas_1 + ff_distance_lower <= ind_op <= ind_meas_1 + ff_distance_upper
                case _:
                    pytest.fail("mode of feedforward parameter does not match.")
    make_figure(g)
