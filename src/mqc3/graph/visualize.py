"""Visualizer of graph representation."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.path import Path as MplPath
from typing_extensions import TypedDict

from mqc3.feedforward import FeedForward
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.program import GraphOpParam, GraphRepr, ModeMeasuredVariable, PosMeasuredVariable
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphOperation as PbOperation

try:
    # Available in Python 3.11 and later
    from typing import NotRequired, Unpack
except ImportError:
    # Use typing_extensions for Python 3.10 and earlier
    from typing_extensions import NotRequired, Unpack  # noqa:UP035

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from mqc3.graph.ops import GraphOpParam
    from mqc3.graph.program import Operation

logger = logging.getLogger(__name__)


def _shrink_graph(graph: GraphRepr) -> GraphRepr:
    max_h = 0
    max_w = 0
    for h in range(graph.n_local_macronodes):
        for w in range(graph.n_steps):
            op = graph.get_operation(h, w)
            if op.type() != PbOperation.OPERATION_TYPE_WIRING or op.swap:
                max_h = max(max_h, h)
                max_w = max(max_w, w)

    shrunk = GraphRepr(max_h + 1, max_w + 1)
    for h in range(max_h + 1):
        for w in range(max_w + 1):
            shrunk.place_operation(graph.get_operation(h, w))

    return shrunk


def _normalize_direction(vec: tuple[float, float]) -> tuple[float, float]:
    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
    if norm == 0:
        return vec
    return (vec[0] / norm, vec[1] / norm)


def replace_nth_format_field(f: str, index: int) -> str:
    """Replaces the nth format field (`{}`, `{:<format>}`, `{!s}`, etc.) in a given format string with `{}`.

    Parameters:
        f (str): The format string.
        theta_index (int): The index (0-based) of the format field to be replaced.

    Returns:
        str: The modified format string with the specified format field replaced.
    """
    # Matches `{}`, `{:.2f}`, `{!s}`, `{!r}`, `{:<10}`, etc.
    pattern = r"(\{[^{}]*\})"
    matches = list(re.finditer(pattern, f))  # Find all format placeholders

    if index >= len(matches):
        return f  # If the index is out of range, return the original string

    new_f = []
    last_end = 0

    for i, match in enumerate(matches):
        start, end = match.span()
        if i == index:
            new_f.append(f[last_end:start] + "{}")  # Replace only the specified format field
        else:
            new_f.append(f[last_end:end])  # Keep other format fields unchanged
        last_end = end

    new_f.append(f[last_end:])  # Append the remaining part of the string
    return "".join(new_f)


def _safe_format(f: str, values: list[float | str]) -> str:
    for i in range(len(values)):
        if isinstance(values[i], str):
            f = replace_nth_format_field(f, i)
    return f.format(*values)


class _Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class _VisualizeConfig:
    title: str = ""
    scale: float = 2.0
    fontsize: float = 10.0

    macronode_radius: float = 1.0
    macronode_edge_color: str = "lightgray"
    micronode_radius: float = 0.2
    micronode_edge_color: str = "lightgray"
    highlight_macronode_edge_color: dict[tuple[int, int], str] = field(default_factory=dict)

    show_op_params: bool = False
    operation_colors: dict[PbOperation.OperationType, str] = field(
        default_factory=lambda: {
            PbOperation.OPERATION_TYPE_MEASUREMENT: "black",
            PbOperation.OPERATION_TYPE_INITIALIZATION: "black",
            PbOperation.OPERATION_TYPE_PHASE_ROTATION: "violet",
            PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT: "cyan",
            PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT: "olive",
            PbOperation.OPERATION_TYPE_SQUEEZING: "teal",
            PbOperation.OPERATION_TYPE_SQUEEZING_45: "teal",
            PbOperation.OPERATION_TYPE_ARBITRARY_FIRST: "darkgray",
            PbOperation.OPERATION_TYPE_ARBITRARY_SECOND: "darkgray",
            PbOperation.OPERATION_TYPE_CONTROLLED_Z: "orange",
            PbOperation.OPERATION_TYPE_BEAM_SPLITTER: "yellow",
            PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR: "brown",
            PbOperation.OPERATION_TYPE_MANUAL: "gray",
            PbOperation.OPERATION_TYPE_WIRING: "white",
        },
    )
    operation_description: dict[PbOperation.OperationType, str] = field(
        default_factory=lambda: {
            PbOperation.OPERATION_TYPE_MEASUREMENT: "$M$",
            PbOperation.OPERATION_TYPE_INITIALIZATION: "$I$",
            PbOperation.OPERATION_TYPE_PHASE_ROTATION: "$R$",
            PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT: "$P$",
            PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT: "$Q$",
            PbOperation.OPERATION_TYPE_SQUEEZING: "$S$",
            PbOperation.OPERATION_TYPE_SQUEEZING_45: "$S_{45}$",
            PbOperation.OPERATION_TYPE_ARBITRARY_FIRST: "$Arb_{1}$",
            PbOperation.OPERATION_TYPE_ARBITRARY_SECOND: "$Arb_{2}$",
            PbOperation.OPERATION_TYPE_CONTROLLED_Z: "$C_Z$",
            PbOperation.OPERATION_TYPE_BEAM_SPLITTER: "$B$",
            PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR: "$P_2$",
            PbOperation.OPERATION_TYPE_MANUAL: "$Man$",
            PbOperation.OPERATION_TYPE_WIRING: "",
        },
    )
    operation_params_format: dict[PbOperation.OperationType, str] = field(
        default_factory=lambda: {
            PbOperation.OPERATION_TYPE_MEASUREMENT: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_INITIALIZATION: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_PHASE_ROTATION: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_SQUEEZING: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_SQUEEZING_45: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_ARBITRARY_FIRST: r"$({:.2f},{:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_ARBITRARY_SECOND: r"$({:.2f},{:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_CONTROLLED_Z: r"$({:.2f})$",
            PbOperation.OPERATION_TYPE_BEAM_SPLITTER: r"$({:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR: r"$({:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_MANUAL: r"$({:.2f},{:.2f},{:.2f},{:.2f})$",
            PbOperation.OPERATION_TYPE_WIRING: "",
        },
    )
    operation_params_symbol_format: dict[tuple[PbOperation.OperationType, int], str] = field(
        default_factory=lambda: {
            (PbOperation.OPERATION_TYPE_MEASUREMENT, 0): r"$\theta$",
            (PbOperation.OPERATION_TYPE_INITIALIZATION, 0): r"$\theta$",
            (PbOperation.OPERATION_TYPE_PHASE_ROTATION, 0): r"$\phi$",
            (PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT, 0): r"$\kappa$",
            (PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT, 0): r"$\eta$",
            (PbOperation.OPERATION_TYPE_SQUEEZING, 0): r"$\theta$",
            (PbOperation.OPERATION_TYPE_SQUEEZING_45, 0): r"$\theta$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_FIRST, 0): r"$\alpha$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_FIRST, 1): r"$\beta$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_FIRST, 2): r"$\lambda$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_SECOND, 0): r"$\alpha$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_SECOND, 1): r"$\beta$",
            (PbOperation.OPERATION_TYPE_ARBITRARY_SECOND, 2): r"$\lambda$",
            (PbOperation.OPERATION_TYPE_CONTROLLED_Z, 0): r"$g$",
            (PbOperation.OPERATION_TYPE_BEAM_SPLITTER, 0): r"$\sqrt{r}$",
            (PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR, 0): r"$\theta_{rel}$",
            (PbOperation.OPERATION_TYPE_MANUAL, 0): r"$\theta_{a}$",
            (PbOperation.OPERATION_TYPE_MANUAL, 1): r"$\theta_{b}$",
            (PbOperation.OPERATION_TYPE_MANUAL, 2): r"$\theta_{c}$",
            (PbOperation.OPERATION_TYPE_MANUAL, 3): r"$\theta_{d}$",
            (PbOperation.OPERATION_TYPE_WIRING, 0): "",
        },
    )

    mode_none_color: str = "lightgray"
    mode_none_linestyle: str = ":"
    mode_colors: list[str] = field(
        default_factory=lambda: [
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "cyan",
            "magenta",
            "yellow",
        ],
    )

    show_displacement: bool = True
    displacement_text_format: str = r"$\binom{{{:.2f}}}{{{:.2f}}}$"

    ignore_wiring: bool = False
    show_mode_index: bool = False

    show_feedforward: bool = True
    show_feedforward_param_label: bool = True
    feedforward_param_fontsize: float = 8.0
    feedforward_arrow_style: str = "->"
    feedforward_line_style: str = "-"
    feedforward_arrow_color: str = "lightgreen"

    def _find_color_undefined_op(
        self,
        graph: GraphRepr,
    ) -> PbOperation.OperationType | None:
        for ind in range(graph.n_total_macronodes):
            op = graph.get_operation(*graph.get_coord(ind))
            if op.type() not in self.operation_colors:
                return op.type()

        return None

    def _find_color_undefined_mode(
        self, graph: GraphRepr, io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]]
    ) -> int:
        ret = -1
        for ind in range(graph.n_total_macronodes):
            for mode in io_modes_dict[graph.get_coord(ind)]:
                if mode >= len(self.mode_colors) and mode > ret:
                    ret = mode

        return ret

    def _find_desc_undefined_op(
        self,
        graph: GraphRepr,
    ) -> PbOperation.OperationType | None:
        for ind in range(graph.n_total_macronodes):
            op = graph.get_operation(*graph.get_coord(ind))
            if op.type() not in self.operation_description:
                return op.type()

        return None

    def verify(self, graph: GraphRepr, io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]]) -> None:
        # check operation colors
        op = self._find_color_undefined_op(graph)
        if op is not None:
            logger.warning(
                "%s is not defined in operation colors",
                op,
            )

        # check mode colors
        mode = self._find_color_undefined_mode(graph, io_modes_dict)
        if mode >= 0:
            logger.warning(
                "Undefined mode colors (demand: %d, defined: %d)",
                mode + 1,
                len(self.mode_colors),
            )

        # check operation descriptions
        op = self._find_desc_undefined_op(graph)
        if op is not None:
            logger.warning(
                "%s is not defined in operation descriptions",
                op,
            )

        # check radius
        if self.micronode_radius * 2 >= self.macronode_radius:
            logger.warning(
                "micronode_radius (%f) should be half of macronode_radius(%f)",
                self.micronode_radius,
                self.macronode_radius,
            )

    def get_macronode_pos(self, hw: tuple[int, int]) -> tuple[float, float]:
        multiplier = self.macronode_radius * 3
        return hw[1] * multiplier, -hw[0] * multiplier

    def get_micronode_pos(
        self,
        macronode_hw: tuple[int, int],
        direction: _Direction,
    ) -> tuple[float, float]:
        macronode_pos = self.get_macronode_pos(macronode_hw)
        length = self.macronode_radius * 0.5

        if direction == _Direction.UP:
            return macronode_pos[0], macronode_pos[1] + length
        if direction == _Direction.DOWN:
            return macronode_pos[0], macronode_pos[1] - length
        if direction == _Direction.LEFT:
            return macronode_pos[0] - length, macronode_pos[1]
        if direction == _Direction.RIGHT:
            return macronode_pos[0] + length, macronode_pos[1]

        msg = f"Invalid direction: {direction}."
        raise ValueError(msg)

    def get_op_color(self, op: PbOperation.OperationType) -> str:
        return self.operation_colors[op]

    def get_op_desc(self, op: PbOperation.OperationType) -> str:
        return self.operation_description[op]

    def get_op_params(self, op: PbOperation.OperationType, params: list[float | str]) -> str:
        return _safe_format(self.operation_params_format[op], params)

    def make_macronode_patch(
        self,
        hw: tuple[int, int],
        op: PbOperation.OperationType,
    ) -> patches.Circle:
        ec = self.macronode_edge_color
        if hw in self.highlight_macronode_edge_color:
            ec = self.highlight_macronode_edge_color[hw]
        return patches.Circle(
            xy=self.get_macronode_pos(hw),
            radius=self.macronode_radius,
            fc=self.get_op_color(op),
            ec=ec,
        )

    def make_micronode_patch(
        self,
        macronode_hw: tuple[int, int],
        direction: _Direction,
        mode: int,
    ) -> patches.Circle:
        return patches.Circle(
            xy=self.get_micronode_pos(macronode_hw, direction),
            radius=self.micronode_radius,
            fc=self.get_mode_color(mode),
            ec=self.micronode_edge_color,
        )

    def add_operation_text(
        self,
        ax: Axes,
        hw: tuple[int, int],
        op: PbOperation.OperationType,
        params: list[float | str],
    ) -> None:
        x, y = self.get_macronode_pos(hw)
        op_desc = self.get_op_desc(op)
        if self.show_op_params:
            op_desc += self.get_op_params(op, params)
        ax.text(
            x + self.macronode_radius,
            y + self.macronode_radius,
            op_desc,
            ha="center",
            fontsize=self.fontsize,
        )

    def add_displacement_text(
        self,
        ax: Axes,
        hw: tuple[int, int],
        op: Operation,
    ) -> None:
        x, y = self.get_macronode_pos(hw)

        epsilon = 1e-3

        # displacement at k-1
        if (
            isinstance(op.displacement_k_minus_1[0], FeedForward)
            or isinstance(op.displacement_k_minus_1[1], FeedForward)
            or abs(op.displacement_k_minus_1[0]) > epsilon
            or abs(op.displacement_k_minus_1[1]) > epsilon
        ):
            ax.text(
                x + self.micronode_radius * 1.1,
                y + self.macronode_radius * 1.5,
                _safe_format(
                    self.displacement_text_format,
                    [
                        d if not isinstance(d, FeedForward) else (r"d_{x}^{-1}", r"d_{p}^{-1}")[i]  # noqa: RUF027
                        for i, d in enumerate(op.displacement_k_minus_1)
                    ],
                ),
                ha="left",
                fontsize=self.fontsize,
            )

        # displacement at k-n
        if (
            isinstance(op.displacement_k_minus_n[0], FeedForward)
            or isinstance(op.displacement_k_minus_n[1], FeedForward)
            or abs(op.displacement_k_minus_n[0]) > epsilon
            or abs(op.displacement_k_minus_n[1]) > epsilon
        ):
            ax.text(
                x - self.macronode_radius * 2,
                y + self.micronode_radius * 1.1,
                _safe_format(
                    self.displacement_text_format,
                    [
                        d if not isinstance(d, FeedForward) else (r"d_{x}^{-N}", r"d_{p}^{-N}")[i]  # noqa: RUF027
                        for i, d in enumerate(op.displacement_k_minus_n)
                    ],
                ),
                ha="left",
                fontsize=self.fontsize,
            )

    def add_mode_index_text(
        self,
        ax: Axes,
        hw: tuple[int, int],
        mode_k1: int,
        mode_kn: int,
    ) -> None:
        x, y = self.get_macronode_pos(hw)

        if mode_k1 != BLANK_MODE:
            ax.text(
                x - self.micronode_radius * 1.1,
                y - self.macronode_radius * 1.5,
                str(mode_k1),
                ha="right",
                fontsize=self.fontsize,
            )

        if mode_kn != BLANK_MODE:
            ax.text(
                x + self.macronode_radius * 1.5,
                y - self.micronode_radius * 1.1,
                str(mode_kn),
                ha="center",
                fontsize=self.fontsize,
            )

    def make_swap_sign_line(self, hw: tuple[int, int]) -> tuple[list[tuple[float, float]], str]:
        x, y = self.get_macronode_pos(hw)
        diff = self.macronode_radius / math.sqrt(2)
        line = [(x - diff, y + diff), (x + diff, y - diff)]
        return line, self.macronode_edge_color

    def make_through_sign_patch_list(
        self,
        hw: tuple[int, int],
    ) -> list[patches.Patch]:
        x, y = self.get_macronode_pos(hw)
        diff = self.macronode_radius / math.sqrt(2)
        epsilon = (self.macronode_radius / 2.0 - self.micronode_radius) / 2.0

        # \_/
        upper = patches.Polygon(
            xy=np.array([
                (x - diff, y + diff),
                (x - epsilon, y + epsilon),
                (x + epsilon, y + epsilon),
                (x + diff, y + diff),
            ]),
            ec=self.macronode_edge_color,
            fc=(0, 0, 0, 0),  # transparent
            closed=False,
        )

        # /¯\
        lower = patches.Polygon(
            xy=np.array([
                (x - diff, y - diff),
                (x - epsilon, y - epsilon),
                (x + epsilon, y - epsilon),
                (x + diff, y - diff),
            ]),
            ec=self.macronode_edge_color,
            fc=(0, 0, 0, 0),  # transparent
            closed=False,
        )
        return [upper, lower]

    def get_mode_color(self, mode: int) -> str:
        if mode < 0:
            return self.mode_none_color

        return self.mode_colors[mode % len(self.mode_colors)]

    def get_mode_linestyle(self, mode: int) -> str:
        return self.mode_none_linestyle if mode < 0 else "-"

    def make_straight_edge_patch(
        self,
        from_hw: tuple[int, int],
        direction: _Direction,
        mode: int,
    ) -> patches.FancyArrowPatch:
        if direction == _Direction.RIGHT:
            from_x, from_y = self.get_micronode_pos(from_hw, _Direction.RIGHT)

            to_hw = (from_hw[0], from_hw[1] + 1)
            to_x, to_y = self.get_micronode_pos(to_hw, _Direction.LEFT)

            a = (from_x + self.micronode_radius, from_y)
            b = (to_x - self.micronode_radius, to_y)
        elif direction == _Direction.DOWN:
            from_x, from_y = self.get_micronode_pos(from_hw, _Direction.DOWN)

            to_hw = (from_hw[0] + 1, from_hw[1])
            to_x, to_y = self.get_micronode_pos(to_hw, _Direction.UP)

            a = (from_x, from_y - self.micronode_radius)
            b = (to_x, to_y + self.micronode_radius)
        else:
            msg = f"No edge from {from_hw} to {direction}."
            raise ValueError(msg)

        color = self.get_mode_color(mode)
        return patches.FancyArrowPatch(
            posA=a,
            posB=b,
            arrowstyle="->",
            linestyle=self.get_mode_linestyle(mode),
            ec=color,
            mutation_scale=10,
        )

    def make_curved_edge_patch(
        self,
        from_hw: tuple[int, int],
        mode: int,
    ) -> patches.FancyArrowPatch:
        from_x, from_y = self.get_micronode_pos(from_hw, _Direction.DOWN)

        to_hw = (0, from_hw[1] + 1)
        to_x, to_y = self.get_micronode_pos(to_hw, _Direction.UP)

        # magic numbers
        small = self.micronode_radius * 1.5
        large = self.macronode_radius * 3

        path = MplPath(
            np.array([
                (from_x, from_y - small),
                ((from_x + to_x * 2) / 3.0, from_y - large),
                ((from_x + to_x) / 2.0, (from_y + to_y) / 2.0),
                ((from_x * 2 + to_x) / 3.0, to_y + large),
                (to_x, to_y + small),
            ]),
            [
                MplPath.MOVETO,
                MplPath.CURVE3,
                MplPath.CURVE3,
                MplPath.CURVE3,
                MplPath.CURVE3,
            ],
        )

        return patches.FancyArrowPatch(
            arrowstyle="->",
            linestyle=self.get_mode_linestyle(mode),
            ec=self.get_mode_color(mode),
            path=path,
            mutation_scale=10,
        )

    def make_feedforward_edge_patch(
        self,
        from_hw: tuple[int, int],
        direction: _Direction,
        to_hw: tuple[int, int],
    ) -> patches.FancyArrowPatch:
        # Edge from micronode to macronode.
        from_x, from_y = self.get_micronode_pos(from_hw, direction)
        to_x, to_y = self.get_macronode_pos(to_hw)

        dir_x, dir_y = _normalize_direction((to_x - from_x, to_y - from_y))

        a = (from_x + self.micronode_radius * dir_x, from_y + self.micronode_radius * dir_y)
        b = (to_x - self.macronode_radius * dir_x, to_y - self.macronode_radius * dir_y)

        return patches.FancyArrowPatch(
            posA=a,
            posB=b,
            arrowstyle=self.feedforward_arrow_style,
            linestyle=self.feedforward_line_style,
            ec=self.feedforward_arrow_color,
            mutation_scale=10,
        )

    def add_feedforward_text(
        self,
        ax: Axes,
        from_hw: tuple[int, int],
        direction: _Direction,
        to_hw: tuple[int, int],
        feedforward_text: str,
    ) -> None:
        # Edge from micronode to macronode.
        from_x, from_y = self.get_micronode_pos(from_hw, direction)
        to_x, to_y = self.get_macronode_pos(to_hw)

        dir_x, dir_y = _normalize_direction((to_x - from_x, to_y - from_y))

        b = (to_x - 0.8 * self.macronode_radius * dir_x, to_y - 0.5 * self.macronode_radius * dir_y)

        ax.text(
            b[0],
            b[1],
            feedforward_text,
            ha="left",
            fontsize=self.feedforward_param_fontsize,
        )


def _plot_macronodes(
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> Axes:
    macronodes_patches = []
    lines = []
    line_colors = []
    for ind in range(graph.n_total_macronodes):
        hw = graph.get_coord(ind)
        op = graph.get_operation(*hw)
        left, up, right, down = io_modes_dict[hw]

        # macronode itself
        macronodes_patches.append(config.make_macronode_patch(hw, op.type()))

        # operation name
        # Print feedforward parameters as nan.
        config.add_operation_text(
            ax,
            hw,
            op.type(),
            [
                p if not isinstance(p, FeedForward) else config.operation_params_symbol_format[op.type(), i].strip("$")
                for i, p in enumerate(op.parameters)
            ],
        )

        # displacement
        if config.show_displacement:
            config.add_displacement_text(ax, hw, op)

        # mode index
        if config.show_mode_index:
            config.add_mode_index_text(ax, hw, down, right)

        # through or swap sign
        is_through = left == right and up == down
        is_swap = left == down and up == right

        if is_swap and not is_through:
            line, color = config.make_swap_sign_line(hw)
            lines.append(line)
            line_colors.append(color)
        elif not is_swap and is_through:
            macronodes_patches.extend(config.make_through_sign_patch_list(hw))
        # if both, might be empty macronode
        # if neither, might be measurement or initialization

    ax.add_collection(PatchCollection(macronodes_patches, match_original=True), autolim=False)
    ax.add_collection(LineCollection(lines, colors=line_colors), autolim=False)

    return ax


def _plot_micronodes(
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> Axes:
    micronode_patches = []
    for ind in range(graph.n_total_macronodes):
        hw = graph.get_coord(ind)
        left, up, right, down = io_modes_dict[hw]

        zipped = [
            (left, _Direction.LEFT),
            (up, _Direction.UP),
            (right, _Direction.RIGHT),
            (down, _Direction.DOWN),
        ]
        micronode_patches.extend(config.make_micronode_patch(hw, direction, mode) for mode, direction in zipped)

    ax.add_collection(PatchCollection(micronode_patches, match_original=True), autolim=False)

    return ax


def _plot_edges(
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> Axes:
    for ind in range(graph.n_total_macronodes):
        hw = graph.get_coord(ind)
        if ind + graph.n_local_macronodes < graph.n_total_macronodes:
            right_mode = io_modes_dict[hw][2]
            ax.add_patch(
                config.make_straight_edge_patch(
                    hw,
                    _Direction.RIGHT,
                    right_mode,
                ),
            )

        if ind + 1 < graph.n_total_macronodes:
            down_mode = io_modes_dict[hw][3]
            if (ind + 1) % graph.n_local_macronodes == 0:
                ax.add_patch(config.make_curved_edge_patch(hw, down_mode))
            else:
                ax.add_patch(
                    config.make_straight_edge_patch(
                        hw,
                        _Direction.DOWN,
                        down_mode,
                    ),
                )
    return ax


def _calc_pos_measured_variable(
    var: ModeMeasuredVariable, graph: GraphRepr, io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]]
) -> PosMeasuredVariable:
    # Same implementation as in to_pos_measured_variable.
    # To avoid calculating io_modes_of_macronode for speed up, we use it in this file.
    op_list = graph.calc_mode_operations(var.mode)
    if not op_list:
        msg = f"No operation found for mode {var.mode}."
        raise ValueError(msg)

    op = op_list[-1]
    h, w = op.macronode
    left, up, _, _ = io_modes_dict[h, w]
    if var.mode == up:
        return PosMeasuredVariable(h, w, 0)
    if var.mode == left:
        return PosMeasuredVariable(h, w, 1)

    msg = f"No measurement found for mode {var.mode}."
    raise ValueError(msg)


def _get_pos_var(
    x: FeedForward[PosMeasuredVariable | ModeMeasuredVariable],
    graph: GraphRepr,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> PosMeasuredVariable:
    var = x.variable
    if isinstance(var, PosMeasuredVariable):
        return var
    if isinstance(var, ModeMeasuredVariable):
        return _calc_pos_measured_variable(var, graph, io_modes_dict)
    msg = f"Unknown type of feedforward variable: {var}\n"
    msg += f"Expected PosMeasuredVariable or ModeMeasuredVariable but got {type(var)}."
    raise TypeError(msg)


def _plot_feedforward_edges(  # noqa: PLR0913, PLR0917
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
    to_hw: tuple[int, int],
    params: Iterable[GraphOpParam],
    label_func: Callable[[int], str],
    ff_texts: dict[tuple[tuple[int, int], _Direction], list[str]],
) -> None:
    for index, param in enumerate(params):
        if isinstance(param, FeedForward):
            pos = _get_pos_var(param, graph, io_modes_dict)
            from_hw = (pos.h, pos.w)
            direction = _Direction.UP if pos.bd == 0 else _Direction.LEFT
            ax.add_artist(config.make_feedforward_edge_patch(from_hw, direction, to_hw))

            ff_texts[from_hw, direction].append(label_func(index))


def _plot_feedforward(
    ax: Axes,
    graph: GraphRepr,
    config: _VisualizeConfig,
    io_modes_dict: dict[tuple[int, int], tuple[int, int, int, int]],
) -> Axes:
    for to_hw, op in graph.operations.items():
        ff_texts = defaultdict(list)

        # plot feedforward to operation parameters
        _plot_feedforward_edges(
            ax,
            graph,
            config,
            io_modes_dict,
            to_hw,
            op.parameters,
            lambda i: config.operation_params_symbol_format[op.type(), i],  # noqa:B023
            ff_texts,
        )

        # plot feedforward to displacement parameters
        _plot_feedforward_edges(
            ax,
            graph,
            config,
            io_modes_dict,
            to_hw,
            op.displacement_k_minus_n,
            lambda i: r"$d_{x}^{-N}$" if i == 0 else r"$d_{p}^{-N}$",
            ff_texts,
        )
        _plot_feedforward_edges(
            ax,
            graph,
            config,
            io_modes_dict,
            to_hw,
            op.displacement_k_minus_1,
            lambda i: r"$d_{x}^{-1}$" if i == 0 else r"$d_{p}^{-1}$",
            ff_texts,
        )

        if config.show_feedforward_param_label:
            for (from_hw, direction), texts in ff_texts.items():
                config.add_feedforward_text(ax, from_hw, direction, to_hw, "FF: " + ",".join(texts))

    return ax


def _init_figure(figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.set_aspect("equal", "datalim")
    ax.autoscale(enable=True)

    return fig, ax


class _VisualizeConfigDict(TypedDict):
    title: NotRequired[str]
    scale: NotRequired[float]
    fontsize: NotRequired[float]

    macronode_radius: NotRequired[float]
    macronode_edge_color: NotRequired[str]
    micronode_radius: NotRequired[float]
    micronode_edge_color: NotRequired[str]
    highlight_macronode_edge_color: NotRequired[dict[tuple[int, int], str]]

    show_op_params: NotRequired[bool]
    operation_colors: NotRequired[dict[PbOperation.OperationType, str]]
    operation_description: NotRequired[dict[PbOperation.OperationType, str]]
    operation_params_format: NotRequired[dict[PbOperation.OperationType, str]]
    operation_params_symbol_format: NotRequired[dict[tuple[PbOperation.OperationType, int], str]]

    mode_none_color: NotRequired[str]
    mode_none_linestyle: NotRequired[str]
    mode_colors: NotRequired[list[str]]

    show_displacement: NotRequired[bool]
    displacement_text_format: NotRequired[str]

    ignore_wiring: NotRequired[bool]
    show_mode_index: NotRequired[bool]

    show_feedforward: NotRequired[bool]
    show_feedforward_param_label: NotRequired[bool]
    feedforward_param_fontsize: NotRequired[float]
    feedforward_arrow_style: NotRequired[str]
    feedforward_line_style: NotRequired[str]
    feedforward_arrow_color: NotRequired[str]


def make_figure(graph: GraphRepr, **kwargs: Unpack[_VisualizeConfigDict]) -> Figure:
    """Show the plot of the graph representation.

    Args:
        graph (GraphRepr): Graph representation.
        kwargs: Keyword arguments for visualizing configuration.

    Returns:
        Figure: Matplotlib figure.

    Example:
        >>> from mqc3.graph import GraphRepr, ops
        >>> from mqc3.graph.visualize import make_figure
        >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
        >>> op = ops.PhaseRotation((1, 2), swap=True, phi=1)
        >>> graph.place_operation(op)
        >>> fig = make_figure(graph)
    """
    config = _VisualizeConfig(**kwargs)
    formatted = _shrink_graph(graph) if config.ignore_wiring else graph

    io_modes_dict = formatted.io_modes_dict()

    config.verify(formatted, io_modes_dict)
    # magic number
    fig, ax = _init_figure((formatted.n_steps * config.scale, formatted.n_local_macronodes * config.scale))

    # Add an invisible patch to adjust the datalim.
    left_top_coord = (-config.macronode_radius, config.macronode_radius)
    right_bottom_coord = np.array([config.macronode_radius, -config.macronode_radius]) + config.get_macronode_pos((
        formatted.n_local_macronodes - 1,
        formatted.n_steps - 1,
    ))
    ax.add_patch(
        patches.Rectangle(
            left_top_coord,
            right_bottom_coord[0] - left_top_coord[0],
            right_bottom_coord[1] - left_top_coord[1],
            fill=False,
            alpha=0,
        )
    )

    _plot_macronodes(ax, formatted, config, io_modes_dict)
    _plot_micronodes(ax, formatted, config, io_modes_dict)
    _plot_edges(ax, formatted, config, io_modes_dict)

    if config.show_feedforward:
        _plot_feedforward(ax, formatted, config, io_modes_dict)
    if config.title:
        fig.suptitle(config.title)

    return fig


def savefig(graph: GraphRepr, filename: str, **kwargs: Unpack[_VisualizeConfigDict]) -> None:
    """Save the plot of the graph representation.

    Args:
        graph (GraphRepr): Graph representation.
        filename (str): Save file name.
        kwargs: Keyword arguments for visualizing configuration.

    Example:
        >>> from mqc3.graph import GraphRepr, ops
        >>> from mqc3.graph.visualize import savefig
        >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
        >>> op = ops.PhaseRotation((1, 2), swap=True, phi=1)
        >>> graph.place_operation(op)
        >>> savefig(graph, "graph.png")
    """
    fig = make_figure(graph, **kwargs)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
