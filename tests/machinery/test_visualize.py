"""Test visualizing for machinery representation."""

from math import pi

import matplotlib.pyplot as plt

from mqc3.feedforward import feedforward
from mqc3.graph import GraphRepr, Wiring
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    ArbitraryFirst,
    BeamSplitter,
    ControlledZ,
    Initialization,
    Measurement,
    PhaseRotation,
    ShearPInvariant,
    ShearXInvariant,
)
from mqc3.machinery import MachineryRepr
from mqc3.machinery.visualize import make_figure


def test_all_through():
    g = GraphRepr(5, 7)
    for i in range(3):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))

    m = MachineryRepr.from_graph_repr(g)
    make_figure(m)
    plt.close("all")


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

    m = MachineryRepr.from_graph_repr(g)
    make_figure(m, readout_edge_color="red")
    plt.close("all")


def test_config():
    g = GraphRepr(5, 6)
    for i in range(3):
        g.place_operation(Initialization((i, 0), 0.0, (BLANK_MODE, i)))

    g.place_operation(PhaseRotation((0, 1), 0.0, displacement_k_minus_1=(1.0, 0.0), swap=False))
    g.place_operation(Wiring((1, 1), swap=True))
    g.place_operation(BeamSplitter((2, 1), 0.0, 0.0, displacement_k_minus_1=(1.0, 2.0), swap=False))
    g.place_operation(ShearPInvariant((3, 1), 0.0, swap=True))
    g.place_operation(Wiring((2, 2), swap=True))
    g.place_operation(BeamSplitter((3, 2), 0.0, 0.0, swap=True))
    g.place_operation(Measurement((0, 2), 0.0))
    g.place_operation(Measurement((4, 2), 0.0))
    g.place_operation(Measurement((3, 4), 0.0))

    m = MachineryRepr.from_graph_repr(g)
    make_figure(
        m,
        title="test_config",
        measurement_color="blue",
        operation_color="green",
        readout_edge_color="red",
        fontsize=3.0,
    )
    plt.close("all")


def test_feedforward():
    g = GraphRepr(3, 4)

    g.place_operation(Initialization((1, 0), 0.0, (BLANK_MODE, 0)))
    g.place_operation(PhaseRotation((1, 1), pi / 2, swap=False, displacement_k_minus_n=(1, -1)))
    g.place_operation(Initialization((0, 2), 0.0, (1, BLANK_MODE)))
    g.place_operation(ControlledZ((1, 2), 1, swap=True))
    g.place_operation(Measurement((1, 3), pi / 2))
    g.place_operation(Measurement((2, 2), 0.0))

    v = g.get_measured_value(1, 3, 1)

    @feedforward
    def f1(x: float) -> float:
        return x + 1

    @feedforward
    def f2(x: float) -> float:
        return x * 2

    g.place_operation(
        ArbitraryFirst(
            (2, 3),
            0,
            0,
            f1(v),
            swap=False,
            displacement_k_minus_1=(f2(v), 0),
            displacement_k_minus_n=(0, 1),
        ),
    )

    v = g.get_mode_measured_value(0)
    g.get_operation(2, 3).displacement_k_minus_n = (0, f2(v))

    m = MachineryRepr.from_graph_repr(g)
    make_figure(m, title="test_feedforward", readout_edge_color="red")
    plt.close("all")
