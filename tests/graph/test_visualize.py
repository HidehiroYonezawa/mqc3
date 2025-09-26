"""Test visualizer of graph representation."""

import random

import matplotlib.pyplot as plt
from numpy import pi

from mqc3.feedforward import feedforward
from mqc3.graph import GraphRepr, Wiring
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    ArbitraryFirst,
    ArbitrarySecond,
    BeamSplitter,
    ControlledZ,
    Initialization,
    Manual,
    Measurement,
    PhaseRotation,
    ShearPInvariant,
    ShearXInvariant,
    Squeezing,
    Squeezing45,
    TwoModeShear,
)
from mqc3.graph.visualize import make_figure, replace_nth_format_field

from .common import make_sample_graph


def test_all_through():
    g = GraphRepr(5, 5)
    for i in range(3):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    make_figure(g)
    make_figure(g, ignore_wiring=True)
    plt.close("all")


def test_wire_random():
    g = GraphRepr(5, 6)
    for i in range(3):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    for i in range(5):
        for j in range(1, 6):
            g.place_operation(Wiring((i, j), swap=random.randint(0, 1) == 0))

    make_figure(g)
    plt.close("all")


def test_sample_graph():
    g = make_sample_graph()

    make_figure(g)
    plt.close("all")


def test_arbitrary_through():
    g = GraphRepr(1, 3)
    for i in range(1):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((0, 2), 0.0, 0.0, 0.0, swap=False))

    make_figure(g)
    plt.close("all")


def test_arbitrary_swap():
    g = GraphRepr(1, 3)
    for i in range(1):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))
    g.place_operation(ArbitraryFirst((0, 1), 0.0, 0.0, 0.0, swap=False))
    g.place_operation(ArbitrarySecond((0, 2), 0.0, 0.0, 0.0, swap=True))

    make_figure(g)
    plt.close("all")


def test_shear():
    g = GraphRepr(3, 4)
    for i in range(3):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))

    # w=1
    g.place_operation(Wiring((0, 1), swap=False))
    g.place_operation(ShearXInvariant((1, 1), kappa=0.0, swap=False))
    g.place_operation(ShearPInvariant((2, 1), eta=0.0, swap=False))
    # w=2
    g.place_operation(TwoModeShear((0, 2), a=0.0, b=0.0, swap=False))
    g.place_operation(TwoModeShear((1, 2), a=0.0, b=0.0, swap=False))
    g.place_operation(TwoModeShear((2, 2), a=0.0, b=0.0, swap=False))
    # w=3
    g.place_operation(Measurement((0, 3), 0.0))
    g.place_operation(Measurement((1, 3), 0.0))
    g.place_operation(Measurement((2, 3), 0.0))

    make_figure(g)
    plt.close("all")


def test_op_params():
    g = GraphRepr(6, 13)
    g.place_operation(Initialization(macronode=(0, 0), theta=pi / 2, initialized_modes=(BLANK_MODE, 0)))
    g.place_operation(Wiring(macronode=(0, 1), swap=False))
    g.place_operation(PhaseRotation(macronode=(0, 2), phi=pi / 4, swap=False))
    g.place_operation(ShearXInvariant(macronode=(0, 3), kappa=-1, swap=False))
    g.place_operation(ShearPInvariant(macronode=(0, 4), eta=2, swap=False))
    g.place_operation(Squeezing(macronode=(0, 5), theta=3 * pi / 2, swap=False))
    g.place_operation(Squeezing45(macronode=(0, 6), theta=5 * pi / 2, swap=False))
    g.place_operation(ArbitraryFirst(macronode=(0, 7), alpha=1, beta=2, lam=3, swap=False))
    g.place_operation(ArbitrarySecond(macronode=(0, 8), alpha=1, beta=2, lam=3, swap=True))

    g.place_operation(Initialization(macronode=(1, 0), theta=pi, initialized_modes=(BLANK_MODE, 1)))
    g.place_operation(ControlledZ(macronode=(1, 8), g=4, swap=False))

    g.place_operation(Wiring(macronode=(2, 8), swap=True))
    g.place_operation(Wiring(macronode=(1, 9), swap=True))
    g.place_operation(BeamSplitter(macronode=(2, 9), sqrt_r=0.1, theta_rel=pi / 8, swap=False))

    g.place_operation(Wiring(macronode=(3, 9), swap=True))
    g.place_operation(Wiring(macronode=(2, 10), swap=True))
    g.place_operation(TwoModeShear(macronode=(3, 10), a=10, b=20, swap=False))

    g.place_operation(Wiring(macronode=(4, 10), swap=True))
    g.place_operation(Wiring(macronode=(3, 11), swap=True))
    g.place_operation(Manual(macronode=(4, 11), theta_a=1, theta_b=2, theta_c=3, theta_d=4, swap=False))

    g.place_operation(Measurement(macronode=(5, 11), theta=0))
    g.place_operation(Measurement(macronode=(4, 12), theta=pi / 2))

    make_figure(g, show_op_params=True, fontsize=5)
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

    make_figure(
        g,
        show_op_params=True,
        fontsize=10,
        feedforward_param_fontsize=10,
        highlight_macronode_edge_color={(2, 3): "red"},
    )
    plt.close("all")


def test_replace_nth():
    f1 = "Value: {:.2f}, String: {!s}, Default: {}"
    f2 = "X({{{:4f}) Y({}) Z({!r})"
    f3 = "Name({!s}) Age({}) Money({:,})"
    f4 = "Test {{{:.2f}}} Normal({})"
    f5 = "Various: {:+d} {:#x} {!a} {:%}"

    assert replace_nth_format_field(f1, 0) == "Value: {}, String: {!s}, Default: {}"
    assert replace_nth_format_field(f1, 1) == "Value: {:.2f}, String: {}, Default: {}"
    assert replace_nth_format_field(f1, 2) == "Value: {:.2f}, String: {!s}, Default: {}"
    assert replace_nth_format_field(f2, 0) == "X({{{}) Y({}) Z({!r})"
    assert replace_nth_format_field(f2, 1) == "X({{{:4f}) Y({}) Z({!r})"
    assert replace_nth_format_field(f2, 2) == "X({{{:4f}) Y({}) Z({})"
    assert replace_nth_format_field(f3, 2) == "Name({!s}) Age({}) Money({})"
    assert replace_nth_format_field(f4, 0) == "Test {{{}}} Normal({})"
    assert replace_nth_format_field(f5, 3) == "Various: {:+d} {:#x} {!a} {}"
