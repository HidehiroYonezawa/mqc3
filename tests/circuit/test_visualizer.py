"""Test visualizer of circuit representation."""


# pyright: reportUnusedExpression=false

from math import pi

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mqc3.circuit import CircuitRepr
from mqc3.circuit.ops import intrinsic, std
from mqc3.circuit.visualize import (
    DEFAULT_INTRINSIC_OP_COLOR_TABLE,
    DEFAULT_OPBOX_VERTICAL_MARGIN,
    DEFAULT_OPBOX_WIDTH,
    CircuitVisualizer,
    OpBox,
    VisualizeConfig,
    make_figure,
)
from mqc3.feedforward import feedforward

default_config = VisualizeConfig()


def sample_circuit_empty() -> CircuitRepr:
    c = CircuitRepr("empty_circuit")
    c.Q([0, 1, 2, 3])
    return c


def sample_circuit() -> CircuitRepr:
    c = CircuitRepr("opbox_init")
    c.Q(0) | intrinsic.Displacement(0.0, 0.0)
    c.Q(1) | intrinsic.Displacement(0.0, 0.0)
    c.Q(0, 1) | intrinsic.BeamSplitter(sqrt_r=1, theta_rel=0.1)
    return c


def sample_circuit_multiple_gates() -> CircuitRepr:
    circuit = CircuitRepr("test_circuit")
    circuit.Q(0) | intrinsic.PhaseRotation(phi=pi) | intrinsic.Displacement(0.0, 0.0)
    circuit.Q(1) | intrinsic.PhaseRotation(phi=0)
    circuit.Q(0, 1) | intrinsic.BeamSplitter(sqrt_r=1, theta_rel=0.1)
    x = circuit.Q(1) | intrinsic.Measurement(theta=0)

    @feedforward
    def f(x: float) -> float:
        return x

    circuit.Q(1, 2) | intrinsic.ControlledZ(g=f(x))
    return circuit


def sample_circuit_with_std_gates() -> CircuitRepr:
    circuit = CircuitRepr("test_circuit")
    circuit.Q(0) | std.Squeezing(r=1.0) | intrinsic.Displacement(0.0, 0.0)
    circuit.Q(1) | intrinsic.PhaseRotation(phi=0)
    circuit.Q(0, 1) | std.BeamSplitter(theta=pi / 2, phi=pi)
    circuit.Q(0, 1) | intrinsic.BeamSplitter(sqrt_r=1, theta_rel=0.1)
    x = circuit.Q(1) | intrinsic.Measurement(theta=0)

    @feedforward
    def f(x: float) -> float:
        return x

    circuit.Q(1, 2) | intrinsic.ControlledZ(g=f(x))
    return circuit


def test_opbox_init():
    opbox = OpBox(1, mode1=0, mode2=0, name="Test", config=default_config)
    assert opbox.column == 1
    assert opbox.mode1 == 0
    assert opbox.mode2 == 0
    assert opbox.name == "Test"
    assert opbox.ff_source_params is None
    assert opbox.is_measurement is False

    opbox = OpBox(
        1,
        mode1=0,
        mode2=0,
        name="Test",
        config=default_config,
        ff_source_params={"alpha": 0, "beta": 1},
        is_measurement=True,
        parameters={"lam": 0.0},
    )
    assert opbox.ff_source_params == {"alpha": 0, "beta": 1}
    assert opbox.is_measurement is True
    assert opbox.parameters == {"lam": 0.0}


def test_opbox_gen_rectangle():
    opbox = OpBox(1, mode1=0, mode2=0, name="BeamSplitter", config=default_config, ff_source_params={"sqrt_r": 1})

    assert opbox.xy == (2, -0.2)
    assert opbox.width == DEFAULT_OPBOX_WIDTH
    assert opbox.height == DEFAULT_OPBOX_VERTICAL_MARGIN * 2

    rect = opbox.gen_rectangle(default_config)
    assert rect.xy == (2, -0.2)
    assert rect.get_width() == DEFAULT_OPBOX_WIDTH
    assert rect.get_height() == DEFAULT_OPBOX_VERTICAL_MARGIN * 2


def test_circuit_visualizer_init():
    circuit = sample_circuit_empty()
    visualizer = CircuitVisualizer(circuit, default_config)

    assert not any(visualizer.op_boxes)
    assert visualizer.max_column == 0
    assert visualizer.num_modes == circuit.n_modes

    circuit = sample_circuit()
    visualizer = CircuitVisualizer(circuit, default_config)
    assert visualizer.max_column == 2
    assert visualizer.num_modes == circuit.n_modes

    circuit = sample_circuit_multiple_gates()
    visualizer = CircuitVisualizer(circuit, default_config)
    assert visualizer.max_column == 5
    assert visualizer.num_modes == circuit.n_modes


def test_make_figure():
    c = sample_circuit_empty()
    fig = make_figure(c)
    assert isinstance(fig, Figure)

    c = sample_circuit()
    fig = make_figure(c)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_ops():
    c = sample_circuit_multiple_gates()
    fig = make_figure(c)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_std_ops():
    c = sample_circuit_with_std_gates()
    fig = make_figure(c)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_make_figure_with_kw():
    c = sample_circuit_multiple_gates()

    op_color_table = DEFAULT_INTRINSIC_OP_COLOR_TABLE.copy()
    op_color_table["Measurement"] = "lightblue"
    fig = make_figure(
        c,
        show_op_legend=True,
        show_parameters=False,
        show_feedforward_param_label=False,
        feedforward_arrow_style="wedge",
        feedforward_arrow_color="blue",
        opbox_width=0.8,
        opbox_vertical_margin=0.1,
        qumode_hline_spacing=1.2,
        intrinsic_operation_color_table=op_color_table,
    )
    assert isinstance(fig, Figure)
    plt.close("all")
