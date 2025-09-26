"""Visualizer for machinery representation."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing_extensions import TypedDict

from mqc3.feedforward import FeedForward
from mqc3.graph import GraphRepr
from mqc3.graph.ops import Manual, Measurement
from mqc3.graph.program import GraphOpParam, PosMeasuredVariable
from mqc3.graph.visualize import make_figure as graph_make_figure
from mqc3.machinery import MachineryRepr
from mqc3.machinery.macronode_angle import MachineOpParam
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphOperation as PbOperation

try:
    # Available in Python 3.11 and later
    from typing import NotRequired, Unpack
except ImportError:
    # Use typing_extensions for Python 3.10 and earlier
    from typing_extensions import NotRequired, Unpack  # noqa:UP035


class _VisualizeConfigDict(TypedDict):
    title: NotRequired[str]
    scale: NotRequired[float]
    fontsize: NotRequired[float]

    macronode_radius: NotRequired[float]
    micronode_radius: NotRequired[float]
    measurement_color: NotRequired[str]
    operation_color: NotRequired[str]
    readout_edge_color: NotRequired[str]

    show_feedforward: NotRequired[bool]
    show_feedforward_param_label: NotRequired[bool]
    feedforward_param_fontsize: NotRequired[float]
    feedforward_arrow_style: NotRequired[str]
    feedforward_line_style: NotRequired[str]
    feedforward_arrow_color: NotRequired[str]


@dataclass
class _VisualizeConfig:
    title: str = ""
    scale: float = 2.0
    fontsize: float = 7.0

    macronode_radius: float = 1.0
    micronode_radius: float = 0.2
    measurement_color: str = "black"
    operation_color: str = "white"
    readout_edge_color: str = "red"

    show_feedforward: bool = True
    show_feedforward_param_label: bool = True
    feedforward_param_fontsize: float = 7.0
    feedforward_arrow_style: str = "->"
    feedforward_line_style: str = "-"
    feedforward_arrow_color: str = "lightgreen"


def _convert_mg_param(g: GraphRepr, param: MachineOpParam) -> GraphOpParam:
    if not isinstance(param, FeedForward):
        return param

    var = param.variable
    coord = g.get_coord(var.macronode_index)
    # abcd (a=0, b=1, c=2, d=3) --> bd (b=0, d=1)
    # a,b --> b   c,d --> d
    bd = 0 if var.abcd in {0, 1} else 1
    return param.func(PosMeasuredVariable(coord[0], coord[1], bd))


def convert_mg(machinery_repr: MachineryRepr) -> GraphRepr:
    """Convert machinery representation to graph representation.

    Args:
        machinery_repr: Machinery representation to convert.

    Returns:
        GraphRepr: Converted graph representation.
    """
    n_local_macronodes = machinery_repr.n_local_macronodes
    n_steps = machinery_repr.n_steps

    g = GraphRepr(n_local_macronodes, n_steps)

    for ind in range(machinery_repr.n_total_macronodes):
        homodyne_angle = machinery_repr.get_homodyne_angle(ind)

        g_displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (
            _convert_mg_param(g, machinery_repr.displacements_k_minus_1[ind][0]),
            _convert_mg_param(g, machinery_repr.displacements_k_minus_1[ind][1]),
        )
        g_displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (
            _convert_mg_param(g, machinery_repr.displacements_k_minus_n[ind][0]),
            _convert_mg_param(g, machinery_repr.displacements_k_minus_n[ind][1]),
        )
        if len(set(homodyne_angle)) == 1:
            # Convert to Measurement operation.
            op = Measurement(
                g.get_coord(ind),
                _convert_mg_param(g, homodyne_angle[0]),
                displacement_k_minus_1=g_displacement_k_minus_1,
                displacement_k_minus_n=g_displacement_k_minus_n,
            )
        else:
            # Convert to Manual operation.
            op = Manual(
                g.get_coord(ind),
                _convert_mg_param(g, homodyne_angle[0]),
                _convert_mg_param(g, homodyne_angle[1]),
                _convert_mg_param(g, homodyne_angle[2]),
                _convert_mg_param(g, homodyne_angle[3]),
                swap=False,
                displacement_k_minus_1=g_displacement_k_minus_1,
                displacement_k_minus_n=g_displacement_k_minus_n,
            )
        g.place_operation(op)

    return g


def make_figure(m: MachineryRepr, **kwargs: Unpack[_VisualizeConfigDict]) -> Figure:
    """Show the plot of the machinery representation.

    Args:
        m (MachineryRepr): Machinery representation.
        kwargs: Keyword arguments for visualizing configuration.

    Returns:
        Figure: Matplotlib figure.

    Example:
        >>> from mqc3.machinery import MachineryRepr
        >>> from mqc3.machinery.visualize import make_figure
        >>> machinery_repr = MachineryRepr(
        ...     n_local_macronodes=1,
        ...     n_steps=3,
        ...     readout_macronode_indices={0, 1, 2},
        ...     displacements_k_minus_1=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
        ...     displacements_k_minus_n=[(0.7, 0.8), (0.9, 1.0), (1.1, 1.2)],
        ... )
        >>> fig = make_figure(machinery_repr, fontsize=6)
    """
    config = _VisualizeConfig(**kwargs)

    # Convert MachineryRepr to GraphRepr.
    g = convert_mg(m)

    operation_colors: dict[PbOperation.OperationType, str] = {
        PbOperation.OPERATION_TYPE_MANUAL: config.operation_color,
        PbOperation.OPERATION_TYPE_MEASUREMENT: config.measurement_color,
    }

    operation_description: dict[PbOperation.OperationType, str] = {
        PbOperation.OPERATION_TYPE_MANUAL: "",
        PbOperation.OPERATION_TYPE_MEASUREMENT: "",
    }

    operation_params_format: dict[PbOperation.OperationType, str] = {
        PbOperation.OPERATION_TYPE_MANUAL: r"$({:.2f},{:.2f},{:.2f},{:.2f})$",
        PbOperation.OPERATION_TYPE_MEASUREMENT: r"${:.2f}$",
    }

    # Change edge color of readout macronodes.
    highlight_macronode_edge_color = {g.get_coord(i): config.readout_edge_color for i in m.readout_macronode_indices}

    fig = graph_make_figure(
        g,
        show_op_params=True,
        scale=config.scale,
        fontsize=config.fontsize,
        macronode_radius=config.macronode_radius,
        micronode_radius=config.micronode_radius,
        operation_colors=operation_colors,
        operation_description=operation_description,
        operation_params_format=operation_params_format,
        highlight_macronode_edge_color=highlight_macronode_edge_color,
        show_feedforward=config.show_feedforward,
        show_feedforward_param_label=config.show_feedforward_param_label,
        feedforward_arrow_style=config.feedforward_arrow_style,
        feedforward_line_style=config.feedforward_line_style,
        feedforward_arrow_color=config.feedforward_arrow_color,
        feedforward_param_fontsize=config.feedforward_param_fontsize,
    )

    if config.title:
        fig.suptitle(config.title)
    return fig


def savefig(m: MachineryRepr, filename: str, **kwargs: Unpack[_VisualizeConfigDict]) -> None:
    """Save the plot of the machinery representation.

    Args:
        m (MachineryRepr): Machinery representation.
        filename (str): Save file name.
        kwargs: Keyword arguments for visualizing configuration.

    Example:
        >>> from mqc3.machinery import MachineryRepr
        >>> from mqc3.machinery.visualize import savefig
        >>> machinery_repr = MachineryRepr(
        ...     n_local_macronodes=3,
        ...     n_steps=3,
        ...     readout_macronode_indices={0, 4, 8},
        ... )
        >>> savefig(machinery_repr, "machinery.png", fontsize=6, readout_edge_color="red")
    """
    fig = make_figure(m, **kwargs)
    fig.savefig(filename)
    plt.close(fig)
