"""Test beamsearch convert algorithm."""

# pyright: reportUnusedExpression=false
import matplotlib.pyplot as plt
import pytest

from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic, std
from mqc3.feedforward import FeedForward, feedforward
from mqc3.graph.convert import BeamSearchConverter, BeamSearchConvertSettings
from mqc3.graph.program import ModeMeasuredVariable
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

    converter = BeamSearchConverter(BeamSearchConvertSettings(n_local_macronodes=10, beam_width=20))
    g = converter.convert(c)
    make_figure(g)
    plt.close("all")


def test_std():
    c = CircuitRepr("test_std")
    c.Q(0) | std.Squeezing(r=0.0)
    c.Q(1, 2) | std.BeamSplitter(theta=0.0, phi=0.0)
    for i in range(3):
        c.Q(i) | intrinsic.Measurement(0.0)

    converter = BeamSearchConverter(BeamSearchConvertSettings(n_local_macronodes=10, beam_width=20))
    g = converter.convert(c)
    make_figure(g)
    plt.close("all")


def test_feedforward():
    ff_distance_lower = 5
    ff_distance_upper = 30

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

    converter = BeamSearchConverter(
        BeamSearchConvertSettings(
            n_local_macronodes=10, beam_width=20, feedforward_distance=(ff_distance_lower, ff_distance_upper)
        )
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
    plt.close("all")
