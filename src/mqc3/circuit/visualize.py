"""Visualizer of circuit representation."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import patches
from typing_extensions import TypedDict

from mqc3.circuit.ops import intrinsic
from mqc3.feedforward import FeedForward

try:
    # Available in Python 3.11 and later
    from typing import NotRequired, Unpack
except ImportError:
    # Use typing_extensions for Python 3.10 and earlier
    from typing_extensions import NotRequired, Unpack  # noqa:UP035

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .program import CircOpParam, CircuitRepr, Operation


logger = logging.getLogger(__name__)

#: OPBOX width
DEFAULT_OPBOX_WIDTH = 1.0
#: OPBOX vertical margin
DEFAULT_OPBOX_VERTICAL_MARGIN = 0.2
#: Interval between mode horizontal lines
DEFAULT_QUMODE_HLINE_SPACING = 1.0
#: Mapping table between intrinsic Operation class names and display names
DEFAULT_INTRINSIC_OP_NAME_TABLE: dict[str, str] = {
    "Measurement": r"$M$",
    "PhaseRotation": r"$R$",
    "ShearXInvariant": r"$P$",
    "ShearPInvariant": r"$Q$",
    "Squeezing": r"$S$",
    "Squeezing45": r"$S_{45}$",
    "Arbitrary": r"$Arb$",
    "ControlledZ": r"$C_Z$",
    "BeamSplitter": r"$B$",
    "TwoModeShear": r"$P_{2}$",
    "Manual": r"$Man$",
    "Displacement": r"$D$",
}
#: Mapping table between user Operation class names and display names
DEFAULT_USER_OP_NAME_TABLE: dict[str, str] = {
    "Squeezing": r"$S_{std}$",
    "BeamSplitter": r"$B_{std}$",
}
#: Mapping table between intrinsic Operation class names and display colors
DEFAULT_INTRINSIC_OP_COLOR_TABLE: dict[str, str] = {
    "Measurement": "white",
    "PhaseRotation": "white",
    "ShearXInvariant": "white",
    "ShearPInvariant": "white",
    "Squeezing": "white",
    "Squeezing45": "white",
    "Arbitrary": "white",
    "ControlledZ": "white",
    "BeamSplitter": "white",
    "TwoModeShear": "white",
    "Manual": "white",
    "Displacement": "white",
}
#: Mapping table between operation parameter names and display names
OPERATION_PARAMETER_FORMAT = {
    "theta": r"$\theta$",
    "x": r"$x$",
    "p": r"$p$",
    "phi": r"$\phi$",
    "kappa": r"$\kappa$",
    "eta": r"$\eta$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "lam": r"$\lambda$",
    "g": r"$g$",
    "sqrt_r": r"$\sqrt{{r}}$",
    "theta_rel": r"$\theta_{{rel}}$",
    "a": r"$a$",
    "b": r"$b$",
    "r": r"$r$",
    "theta_a": r"$\theta_{{a}}$",
    "theta_b": r"$\theta_{{b}}$",
    "theta_c": r"$\theta_{{c}}$",
    "theta_d": r"$\theta_{{d}}$",
}


class _VisualizeConfigDict(TypedDict):
    show_parameters: NotRequired[bool]
    parameters_fontsize: NotRequired[float]

    show_feedforward_param_label: NotRequired[bool]
    feedforward_param_fontsize: NotRequired[float]
    feedforward_arrow_style: NotRequired[str]
    feedforward_arrow_color: NotRequired[str]

    qumode_labels: NotRequired[list[str]]
    qumode_hline_spacing: NotRequired[float]

    intrinsic_operation_name_table: NotRequired[dict[str, str]]
    user_operation_name_table: NotRequired[dict[str, str]]
    intrinsic_operation_color_table: NotRequired[dict[str, str]]

    opbox_width: NotRequired[float]
    opbox_vertical_margin: NotRequired[float]

    show_op_legend: NotRequired[bool]


@dataclass
class VisualizeConfig:
    """Visualizer configuration."""

    show_parameters: bool = True
    parameters_fontsize: float = 8.0

    show_feedforward_param_label: bool = True
    feedforward_param_fontsize: float = 8.0
    feedforward_arrow_style: str = "->"
    feedforward_arrow_color: str = "lightgreen"

    qumode_labels: list[str] | None = None
    qumode_hline_spacing: float = DEFAULT_QUMODE_HLINE_SPACING

    intrinsic_operation_name_table: dict[str, str] = field(default_factory=lambda: DEFAULT_INTRINSIC_OP_NAME_TABLE)
    user_operation_name_table: dict[str, str] = field(default_factory=lambda: DEFAULT_USER_OP_NAME_TABLE)
    intrinsic_operation_color_table: dict[str, str] = field(default_factory=lambda: DEFAULT_INTRINSIC_OP_COLOR_TABLE)

    opbox_width: float = DEFAULT_OPBOX_WIDTH
    opbox_vertical_margin: float = DEFAULT_OPBOX_VERTICAL_MARGIN

    show_op_legend: bool = False


def _get_op_param_names(op: Operation) -> list[str]:
    sig = inspect.signature(op.__init__)
    param_names = list(sig.parameters.keys())
    if param_names and param_names[0] == "self":
        param_names = param_names[1:]
    return param_names


def _get_ff_source_mode(op: Operation) -> dict[str, int]:
    source_ops: dict[str, int] = {}

    for p_name, p_val in zip(_get_op_param_names(op), op.parameters(), strict=False):
        if isinstance(p_val, FeedForward):
            source_mode = p_val.variable.get_from_operation().opnd().get_ids()[0]
            source_ops[p_name] = source_mode

    if len(source_ops) == 0:
        msg = "No FeedForward source operation found."
        logger.error(msg)
        raise ValueError(msg)
    return source_ops


class CircuitVisualizer:
    """Visualizer of circuit representation."""

    def __init__(self, circuit: CircuitRepr, config: VisualizeConfig) -> None:
        """Initialize the CircuitVisualizer object."""
        self.num_modes = circuit.n_modes
        self.config = config

        # Arrange based on operation dependencies
        self.op_boxes: list[list[OpBox | Unplaceable]] = self._schedule_ops(circuit)

        # Get the maximum number of columns
        self.max_column = 0
        if any(self.op_boxes):  # If at least one element in self.op_boxes contains data
            self.max_column = max(mode_op_boxes[-1].column + 1 for mode_op_boxes in self.op_boxes if mode_op_boxes)

    def _schedule_ops(self, circuit: CircuitRepr) -> list[list[OpBox | Unplaceable]]:
        op_box_list: list[list[OpBox | Unplaceable]] = [[] for _ in range(self.num_modes)]
        available_time = [0] * self.num_modes

        for i in range(circuit.n_operations):
            op = circuit.get_operation(i)
            operands: list[int] = op.opnd().get_ids()

            # Convert Operation.parameters to dict format
            param_names = _get_op_param_names(op)
            assert len(param_names) == len(op.parameters())  # noqa: S101
            params_dict: dict[str, CircOpParam] = dict(zip(param_names, op.parameters(), strict=False))

            # Store FF sources as {"ff_param_name": "source_mode_id"}, or None if FF is absent
            ff_source_ops: dict[str, int] | None = None
            start_time = 0

            # If FF exists, place it after the latest qumode in the source Measurement
            if op.has_feedforward_param():
                ff_source_ops = _get_ff_source_mode(op)
                for source_mode_id in ff_source_ops.values():
                    start_time = max(start_time, available_time[source_mode_id])

            # Single-mode gate
            if len(operands) == 1:
                mode = operands[0]
                start_time = max(start_time, available_time[mode])
                available_time[mode] = start_time + 1

                op_box_list[mode].append(
                    OpBox(
                        start_time,
                        mode,
                        mode,
                        name=type(op).__name__,
                        config=self.config,
                        ff_source_params=ff_source_ops,
                        is_measurement=isinstance(op, intrinsic.Measurement),
                        parameters=params_dict,
                        operation=op,
                    ),
                )

            # Two-mode gate
            elif len(operands) == 2:  # noqa:PLR2004
                mode1, mode2 = sorted(operands)

                # Align with the latest time among all modes between mode1 and mode2
                start_time = max(start_time, *(available_time[mode] for mode in range(mode1, mode2 + 1)))

                for mode in range(mode1, mode2 + 1):
                    available_time[mode] = start_time + 1

                    if mode == mode1:
                        # Place the two-mode OpBox only in mode1
                        op_box_list[mode].append(
                            OpBox(
                                start_time,
                                mode1,
                                mode2,
                                type(op).__name__,
                                config=self.config,
                                ff_source_params=ff_source_ops,
                                parameters=params_dict,
                                operation=op,
                            ),
                        )
                    else:
                        op_box_list[mode].append(Unplaceable(start_time, mode2))

        return op_box_list

    def _plot_mode_lines(self, ax: Axes, xlim: tuple[float, float]) -> None:
        # Draw qumodes using hline
        for i in range(self.num_modes):
            y = i
            ax.hlines(self.config.qumode_hline_spacing * y, xlim[0], xlim[1], color="black", linewidth=0.5)

        # Set labels
        ax.set_yticks([self.config.qumode_hline_spacing * y for y in range(self.num_modes)])

        labels = [f"$q_{i}$" for i in range(self.num_modes)]
        if self.config.qumode_labels is not None:
            if len(self.config.qumode_labels) != self.num_modes:
                msg = "The number of qumode labels does not match the number of qumodes. \
                    len(qumode_labels)={len(self.config.qumode_labels)}, num_modes={self.num_modes}}"
                raise ValueError(msg)
            labels = self.config.qumode_labels

        ax.set_yticklabels(labels)

    def _plot_op_boxes(self, ax: Axes) -> None:
        for op_box in (op_box for mode_op_boxes in self.op_boxes for op_box in mode_op_boxes):
            if isinstance(op_box, Unplaceable):
                continue

            rect = op_box.gen_rectangle(self.config)
            ax.add_patch(rect)

            # Draw the operation name
            if isinstance(op_box.op, intrinsic.Intrinsic):
                op_label = self.config.intrinsic_operation_name_table.get(op_box.name, op_box.name)
            else:
                op_label = self.config.user_operation_name_table.get(op_box.name, op_box.name)

            ax.text(
                rect.xy[0] + rect.get_width() / 2,
                rect.xy[1] + rect.get_height() / 2,
                op_label,
                ha="center",
                va="center",
            )

            # Draw arrows if FF exists
            if op_box.ff_source_params is not None:
                self._plot_ff_arrow(ax, op_box)

            # Draw parameters
            if self.config.show_parameters and op_box.parameters is not None:
                label_x = rect.xy[0] + rect.get_width() / 2
                label_y = rect.xy[1] - 0.2

                if op_box.name == "Manual":
                    # Manual does not support parameter display
                    continue

                for k, v in reversed(list(op_box.parameters.items())):
                    if not isinstance(v, FeedForward):
                        text = f"{OPERATION_PARAMETER_FORMAT.get(k, k)}: {v:.2f}"
                        ax.text(
                            label_x,
                            label_y,
                            text,
                            ha="center",
                            va="center",
                            fontsize=self.config.parameters_fontsize,
                        )
                        label_y -= 0.2
                    else:
                        continue

        if self.config.show_op_legend:
            op_box_name_set = {
                op_box.name for mode_op_boxes in self.op_boxes for op_box in mode_op_boxes if isinstance(op_box, OpBox)
            }
            legend_handles = [
                patches.Patch(
                    color=self.config.intrinsic_operation_color_table[name],
                    label=f"{self.config.intrinsic_operation_name_table[name]}: {name}",
                )
                for name in op_box_name_set
            ]

            # Add legend
            ax.legend(handles=legend_handles, handlelength=0, loc="upper right")

    def _plot_ff_arrow(self, ax: Axes, target_opbox: OpBox) -> None:
        assert target_opbox.ff_source_params is not None  # noqa: S101

        # FF sources are stored in OpBox's ff_source_params
        # Stored as {"ff_param_name": "source_mode_id"}, or None if FF is absent
        # Retrieve the source mode for each FF parameter
        for ff_param_name, source_mode_id in target_opbox.ff_source_params.items():
            # Retrieve the MeasurementOpBox placed in source_mode
            source_opbox = next(
                (
                    op_box
                    for op_box in self.op_boxes[source_mode_id]
                    if isinstance(op_box, OpBox) and op_box.is_measurement
                ),
                None,
            )
            if source_opbox is None:
                msg = f"MeasurementOpBox not found for {ff_param_name=}: {source_mode_id=}."
                logger.error(msg)
                raise ValueError(msg)

            # Extend the arrow from the center of the rectangle
            source_opbox_xy = source_opbox.xy
            source_opbox_wh = source_opbox.width, source_opbox.height
            xy_source = (
                source_opbox_xy[0] + source_opbox_wh[0],
                source_opbox_xy[1] + source_opbox_wh[1] / 2,
            )
            target_opbox_xy = target_opbox.xy
            target_opbox_wh = target_opbox.width, target_opbox.height
            xy_target = (
                target_opbox_xy[0],
                target_opbox_xy[1] + target_opbox_wh[1] / 2,
            )

            ax.annotate(
                "",
                xy=xy_target,
                xytext=xy_source,
                arrowprops={
                    "arrowstyle": self.config.feedforward_arrow_style,
                    "color": self.config.feedforward_arrow_color,
                },
            )

            # Place the label at the center of the arrow
            if self.config.show_feedforward_param_label:
                mid_xy = ((xy_source[0] + xy_target[0]) / 2, (xy_source[1] + xy_target[1]) / 2)
                # ff_param_label = f"{ff_param_name}"
                ff_param_label = f"{OPERATION_PARAMETER_FORMAT.get(ff_param_name, ff_param_name)}"
                ax.text(
                    mid_xy[0],
                    mid_xy[1],
                    ff_param_label,
                    ha="center",
                    va="center",
                    fontsize=self.config.feedforward_param_fontsize,
                )

    def make_figure(self) -> Figure:
        """Show the plot of the circuit representation.

        Returns:
            Figure: Matplotlib Figure object.
        """
        fig = plt.figure()
        xlim = (-0.5, 2 * self.max_column + 0.5)
        ylim = (-self.config.qumode_hline_spacing, self.config.qumode_hline_spacing * self.num_modes)

        axes_mode = fig.add_subplot(111, xlim=xlim, ylim=ylim, xticks=[])
        axes_mode.invert_yaxis()
        self._plot_mode_lines(axes_mode, xlim)

        # Add a layer to draw OpBox
        if self.max_column > 0:
            axes_op = fig.add_subplot(111, xlim=xlim, ylim=ylim, facecolor="none", xticks=[], yticks=[])
            axes_op.invert_yaxis()
            self._plot_op_boxes(axes_op)

        return fig


# Class indicating unplaceable locations
class Unplaceable:
    """Unplaceable box for visualizing the circuit representation."""

    def __init__(self, column: int, mode: int) -> None:
        """Initialize the Unplaceable box object."""
        self.column = column
        self.mode = mode

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"Unplaceable({self.column=}, {self.mode=})"


class OpBox:
    """Operation box for visualizing the circuit representation."""

    def __init__(  # noqa: PLR0913
        self,
        column: int,
        mode1: int,
        mode2: int,
        name: str,
        config: VisualizeConfig,
        *,
        ff_source_params: dict[str, int] | None = None,
        is_measurement: bool = False,
        parameters: dict[str, CircOpParam] | None = None,
        operation: Operation | None = None,
    ) -> None:
        """Initialize the Operation box object.

        Raises:
            ValueError: If the operation is not a single-mode or two-mode operation.
        """
        if mode1 > mode2:
            msg = "`mode1` must be less than or equal to `mode2`."
            raise ValueError(msg)

        self.mode1: int = mode1
        self.mode2: int = mode2
        self.column: int = column
        self.name = name
        self.ff_source_params = ff_source_params  # Store the sources if FF exists
        self.is_measurement = is_measurement
        self.parameters = parameters
        self.op = operation

        # Compute xy and wh from VisualizeConfig (config is not retained since it is not used after initialization)
        self.xy = self.column * 2, self.mode1 * config.qumode_hline_spacing - 0.2
        self.width = config.opbox_width
        self.height = config.qumode_hline_spacing * (self.mode2 - self.mode1) + config.opbox_vertical_margin * 2

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"OpBox({self.column=}, {self.mode1=}, {self.mode2=}, {self.name=}, {self.ff_source_params=})"

    def gen_rectangle(self, config: VisualizeConfig) -> patches.Rectangle:
        """Return a rectangle patch for the operation."""
        return patches.Rectangle(
            self.xy,
            self.width,
            self.height,
            edgecolor="blue",
            facecolor=config.intrinsic_operation_color_table[self.name],
        )


def make_figure(circuit: CircuitRepr, **kwargs: Unpack[_VisualizeConfigDict]) -> Figure:
    """Show the plot of the circuit representation.

    Args:
        circuit (CircuitRepr): Circuit representation object.
        kwargs: Keyword arguments for visualizing configuration.

    Returns:
        Figure: Matplotlib Figure object.

    Example:
        >>> from math import pi
        >>> from mqc3.circuit.ops import intrinsic
        >>> from mqc3.circuit.program import CircuitRepr
        >>> from mqc3.circuit.visualize import make_figure
        >>> c = CircuitRepr("sample_circuit")
        >>> c.Q(0) | intrinsic.PhaseRotation(pi) | intrinsic.Measurement(theta=0.0)
        Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
        >>> fig = make_figure(c, show_parameters=True)
    """
    config = VisualizeConfig(**kwargs)
    visualizer = CircuitVisualizer(circuit, config)
    fig = visualizer.make_figure()

    if circuit.name:
        fig.suptitle(circuit.name)

    return fig


def savefig(circuit: CircuitRepr, filename: str, **kwargs: Unpack[_VisualizeConfigDict]) -> None:
    """Save the plot of the circuit representation.

    Args:
        circuit (CircuitRepr): Circuit representation.
        filename (str): Save file name.
        kwargs: Keyword arguments for visualizing configuration.

    Example:
        >>> from math import pi
        >>> from mqc3.circuit.ops import intrinsic
        >>> from mqc3.circuit.program import CircuitRepr
        >>> from mqc3.circuit.visualize import savefig
        >>> c = CircuitRepr("sample_circuit")
        >>> c.Q(0) | intrinsic.PhaseRotation(pi) | intrinsic.Measurement(theta=0.0)
        Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
        >>> savefig(c, "circuit.png", show_parameters=True)
    """
    fig = make_figure(circuit, **kwargs)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
