"""Functions to create a graph representation that includes multiple graphs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from math import pi

from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import Measurement
from mqc3.graph.program import GraphRepr
from mqc3.graph.result import GraphMacronodeMeasuredValue, GraphResult, GraphShotMeasuredValue


@dataclass
class MappingInfo:
    """Mapping between a graph and a composite graph composing multiple shots.

    This class represents mapping information between a graph representation and
    a composite graph representation in which multiple graphs (shots) are composed.
    """

    n_shots: int
    """The number of shots in the composite graph."""

    map: dict[int, tuple[int, int]] = field(default_factory=dict)
    """Dictionary to map a macronode index in the composite graph to a tuple of
    (the index of shot in the composite graph, the macronode index in it)."""


@dataclass
class ComposeSettings:
    """Settings for composing multiple shots into a composite graph."""

    n_shots: int
    """The number of shots to execute the original graph."""

    n_local_macronodes: int
    """The number of macronodes per step in the composite graph."""

    max_steps: int
    """The largest possible number of steps per composite graph."""


@dataclass
class ComposeInfo:
    """Information for returning the results using the composite graph in the original format specified by the user."""

    settings: ComposeSettings
    """ComposeSetting object"""

    map_info: MappingInfo
    """MappingInfo object"""

    original_graph: GraphRepr
    """GraphRepr object input by the user prior to composing"""


def compose_into_composite_graph(  # noqa: C901, PLR0912, PLR0914
    original_graph: GraphRepr, settings: ComposeSettings
) -> tuple[GraphRepr, MappingInfo, int]:
    r"""Compose multiple shots into a composite graph.

    The original graph is composed in the composite graph to the maximum extent,
    with the column size equal to `settings.n_local_macronodes` and the row size limited to
    `settings.max_steps`.

    Original graphs, where at least one mode other than blank mode is passed from the bottom to the top
    (referred to as wrapped graphs), are not arranged vertically and can only be placed horizontally.
    Below each composed original graph, a row of :class:`~mqc3.graph.ops.Measurement` `(theta=pi/2, readout=False)`
    is placed, unless the original graph's size is `settings.n_local_macronodes` or the original graph is not wrapped.
    Similarly, the original graphs arranged horizontally are separated by columns of the same
    `Measurement` macronodes.

    The width of the composite graph is determined by the number of composed graphs,
    up to `settings.max_steps`, after composing them as fully as possible.
    The height is set to `settings.n_local_macronodes`, with the same `Measurement` macronodes
    placed to fill any remaining space.

    Args:
        original_graph (GraphRepr): Original graph to be composed, representing a shot.
        settings (ComposeSettings): Configuration.

    Returns:
        tuple[GraphRepr, MappingInfo, int]: Tuple of (resulting composite graph, `MappingInfo` object,
            the number of composite graph shots required to achieve `settings.n_shots`).

    Raises:
        ValueError: If the original graph has no size.
    """
    map_info: dict[int, tuple[int, int]] = {}

    if original_graph.n_local_macronodes == 0 or original_graph.n_steps == 0:
        msg = "The original graph size must not be 0."
        raise ValueError(msg)

    # number of macronode columns in a small graph
    small_w = original_graph.n_steps

    # whether there is any mode other than blank mode
    # that goes from the bottom to the top of the next column in the original graph
    has_wrap: bool = any(
        original_graph.calc_io_of_macronode(original_graph.n_local_macronodes - 1, w)[3] != BLANK_MODE
        for w in range(original_graph.n_steps)
    )

    # define `small_h`: number of macronode rows in a small graph
    small_h = original_graph.n_local_macronodes

    # number of small graphs in a large graph per column
    h_graphs = min(
        (1 if small_h == settings.n_local_macronodes or has_wrap else settings.n_local_macronodes // (small_h + 1)),
        settings.n_shots,
    )

    # number of small graphs in a large graph per row
    w_graphs = min(
        # the largest possible number of shots are in the composite graph
        (settings.max_steps + 1) // (small_w + 1),
        # the number of shots in the composite graph is below capacity
        (settings.n_shots + h_graphs - 1) // h_graphs,
    )

    # number of macronode rows in a large graph
    large_h = small_h if small_h == settings.n_local_macronodes or has_wrap else (small_h + 1) * h_graphs

    # number of macronode columns in a large graph
    large_w = (small_w + 1) * w_graphs - 1

    # maximum mode index in the original graph
    max_index = max(max(original_graph.calc_io_of_macronode(h, w)) for h, w in product(range(small_h), range(small_w)))

    # create the composite graph
    g_composite = GraphRepr(
        n_local_macronodes=settings.n_local_macronodes,
        n_steps=large_w,
    )

    # loop for each small graph
    i_shot = 0
    for i_w_graph in range(w_graphs):
        for i_h_graph in range(h_graphs):
            if i_shot >= settings.n_shots:
                break
            h_base = (small_h + 1) * i_h_graph
            w_base = (small_w + 1) * i_w_graph

            # place operation for each macronode
            for i_mac_w in range(small_w):
                for i_mac_h in range(small_h):
                    h = h_base + i_mac_h
                    w = w_base + i_mac_w
                    map_info[settings.n_local_macronodes * w + h] = (i_shot, small_h * i_mac_w + i_mac_h)
                    op = deepcopy(original_graph.get_operation(i_mac_h, i_mac_w))
                    op.macronode = (h_base + op.macronode[0], w_base + op.macronode[1])
                    op.initialized_modes = [
                        ((max_index + 1) * (h_graphs * i_w_graph + i_h_graph) + i_mode) if i_mode >= 0 else i_mode
                        for i_mode in op.initialized_modes
                    ]
                    g_composite.place_operation(op)

            # place a measurements raw just below the small graph
            if not (small_h == settings.n_local_macronodes or has_wrap):
                for i_mac_w in range(small_w):
                    h = h_base + small_h
                    w = w_base + i_mac_w
                    g_composite.place_operation(Measurement((h, w), theta=pi / 2, readout=False))

            i_shot += 1
        else:
            # place a measurements column just to the right of the column of the small graphs
            if i_w_graph < w_graphs - 1:
                for i_mac_h in range(large_h):
                    w = w_base + small_w
                    g_composite.place_operation(
                        Measurement((i_mac_h, w), theta=pi / 2, readout=False),
                    )
            continue
        break

    # place measurements at the bottom margin
    if not (small_h == settings.n_local_macronodes or has_wrap):
        for i_mac_w in range(large_w):
            for i_mac_h in range(large_h, settings.n_local_macronodes):
                g_composite.place_operation(
                    Measurement((i_mac_h, i_mac_w), theta=pi / 2, readout=False),
                )

    return (
        g_composite,
        MappingInfo(n_shots=i_shot, map=map_info),
        (settings.n_shots + i_shot - 1) // i_shot,
    )


def decompose_composite_graph_result(
    composite_graph_result: GraphResult,
    map_info: MappingInfo,
    original_graph: GraphRepr,
) -> GraphResult:
    """Convert the execution result of the composite graph back into the execution results of the original graphs.

    Args:
        composite_graph_result (GraphResult): GraphResult object of the composite graph.
        map_info (MappingInfo): MappingInfo object.
        original_graph (GraphRepr): Original graph representation.

    Returns:
        GraphResult: Original graph result.
    """
    small_graph_result = GraphResult(original_graph.n_local_macronodes, [])
    for smv in composite_graph_result:
        small_smvs: dict[int, GraphShotMeasuredValue] = {}
        for mmv in smv:
            if mmv.index not in map_info.map:
                continue
            i_shot, macro_idx = map_info.map[mmv.index]
            small_mmv = GraphMacronodeMeasuredValue(
                macro_idx,
                *original_graph.get_coord(macro_idx),
                m_b=mmv.m_b,
                m_d=mmv.m_d,
            )
            if i_shot not in small_smvs:
                small_smvs[i_shot] = GraphShotMeasuredValue({}, n_local_macronodes=original_graph.n_local_macronodes)
            small_smvs[i_shot].items[small_mmv.index] = small_mmv
        small_graph_result.measured_vals.extend(list(small_smvs.values()))
    return small_graph_result
