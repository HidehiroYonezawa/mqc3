"""Computation result in circuit representation."""

from __future__ import annotations

from collections import OrderedDict, UserList
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsIndex

from mqc3.pb.io import ProtoFormat, load, save
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitResult as PbCircuitResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from mqc3.graph.program import GraphRepr
    from mqc3.graph.result import GraphResult


@dataclass(frozen=True)
class CircuitOperationMeasuredValue:
    """Measured value of a mode specified by the index."""

    index: int
    "Index of the measured mode"

    value: float
    "Measured value"


class CircuitShotMeasuredValue(UserList[CircuitOperationMeasuredValue]):
    """Container of measured values."""

    def __init__(self, initlist: Iterable[CircuitOperationMeasuredValue] | None = None) -> None:
        """Initializes a CircuitShotMeasuredValue container.

        Args:
            initlist (list[CircuitOperationMeasuredValue] | None):
                A list of `CircuitOperationMeasuredValue` instances to initialize the container.
                Defaults to an empty list if not provided.
        """
        super().__init__(initlist or [])

    def get_value(self, index: int) -> float | None:
        """Get the measured value of the specified mode.

        Args:
            index (int): Index of the measured mode.

        Returns:
            float: Measured value.
        """
        return next((item.value for item in self if item.index == index), None)

    def n_operations(self) -> int:
        """Get number of elements of the list.

        Returns:
            int: The number of elements of the list

        Example:
            >>> from mqc3.circuit.result import CircuitShotMeasuredValue, CircuitOperationMeasuredValue
            >>> smv = CircuitShotMeasuredValue([
            ...     CircuitOperationMeasuredValue(index=0, value=1.0),
            ...     CircuitOperationMeasuredValue(index=1, value=0.0),
            ... ])
            >>> smv.n_operations()
            2
        """
        return len(self)


class CircuitResult:
    """The measurement result in circuit representation."""

    def __init__(self, shot_measured_values: Iterable[CircuitShotMeasuredValue]) -> None:
        """Constructor.

        Args:
            shot_measured_values (Iterable[CircuitShotMeasuredValue]): Iterable of CircuitShotMeasuredValue objects

        Raises:
            TypeError: If input argument is not Iterable[CircuitShotMeasuredValue]
        """
        if not all(isinstance(x, CircuitShotMeasuredValue) for x in shot_measured_values):
            message = "The argument type must be Iterable[CircuitShotMeasuredValue]."
            raise TypeError(message)
        self.measured_vals: list[CircuitShotMeasuredValue] = list(shot_measured_values)

    def __iter__(self) -> Iterator:
        """Iterator of `measured_vals` attribute.

        Yields:
            Iterator: Iterator of `measured_vals` attribute

        Example:
            >>> from mqc3.circuit import CircuitResult
            >>> from mqc3.circuit.result import CircuitShotMeasuredValue, CircuitOperationMeasuredValue
            >>> result = CircuitResult([
            ...     CircuitShotMeasuredValue([
            ...         CircuitOperationMeasuredValue(index=0, value=1.0),
            ...         CircuitOperationMeasuredValue(index=1, value=0.0),
            ...     ]),
            ...     CircuitShotMeasuredValue([
            ...         CircuitOperationMeasuredValue(index=0, value=0.0),
            ...         CircuitOperationMeasuredValue(index=1, value=1.0),
            ...     ]),
            ... ])
            >>> for shot_measured_value in result:
            ...     for operation_measured_value in shot_measured_value:
            ...         print(operation_measured_value)
            CircuitOperationMeasuredValue(index=0, value=1.0)
            CircuitOperationMeasuredValue(index=1, value=0.0)
            CircuitOperationMeasuredValue(index=0, value=0.0)
            CircuitOperationMeasuredValue(index=1, value=1.0)
        """
        yield from self.measured_vals

    def __getitem__(
        self,
        index: int | slice | SupportsIndex,
    ) -> CircuitShotMeasuredValue | Sequence[CircuitShotMeasuredValue]:
        """Get the measured value of the shot at the index.

        Args:
            index (int | slice | SupportsIndex): Index or slice.

        Returns:
            CircuitShotMeasuredValue | Sequence[CircuitShotMeasuredValue]: Measured value or sequence of measured
                values
        """
        return self.measured_vals[index]

    def __len__(self) -> int:
        """The number of shots.

        Returns:
            int: The number of shots
        """
        return len(self.measured_vals)

    @staticmethod
    def construct_from_graph_result(graph_result: GraphResult, graph_input: GraphRepr) -> CircuitResult:
        """Construct and return the result object from the graph representation.

        Information from the input in graph representation is also used.

        Args:
            graph_result (GraphResult): Result object in graph representation
            graph_input (GraphRepr): Input in graph representation

        Returns:
            CircuitResult: The result object
        """
        shot_measured_values: list[CircuitShotMeasuredValue] = []
        for graph_smv in graph_result.measured_vals:
            d_mode_measured_val: dict[int, float] = {}
            for graph_mmv in graph_smv:
                h, w = graph_input.get_coord(graph_mmv.index)
                if not graph_input.is_measurement(h, w):
                    continue
                io_modes = graph_input.calc_io_of_macronode(h, w)
                mode_b, mode_d = io_modes[1], io_modes[0]
                if mode_b >= 0:
                    d_mode_measured_val[mode_b] = graph_mmv.m_b
                if mode_d >= 0:
                    d_mode_measured_val[mode_d] = graph_mmv.m_d
                # NOTE: Assuming more than one measurements are not executed per mode
            ordered_d_mode_measured_val = OrderedDict(sorted(d_mode_measured_val.items()))
            shot_measured_values.append(
                CircuitShotMeasuredValue(
                    CircuitOperationMeasuredValue(index=idx, value=val)
                    for idx, val in ordered_d_mode_measured_val.items()
                ),
            )
        return CircuitResult(shot_measured_values)

    def n_shots(self) -> int:
        """Get number of shots in `measured_vals` attribute.

        Returns:
            int: The number of shots in `measured_vals` attribute

        Example:
            >>> from mqc3.circuit import CircuitResult
            >>> from mqc3.circuit.result import CircuitShotMeasuredValue, CircuitOperationMeasuredValue
            >>> result = CircuitResult([
            ...     CircuitShotMeasuredValue([
            ...         CircuitOperationMeasuredValue(index=0, value=1.0),
            ...         CircuitOperationMeasuredValue(index=1, value=0.0),
            ...     ]),
            ...     CircuitShotMeasuredValue([
            ...         CircuitOperationMeasuredValue(index=0, value=0.0),
            ...         CircuitOperationMeasuredValue(index=1, value=1.0),
            ...     ]),
            ... ])
            >>> result.n_shots()
            2
        """
        return len(self.measured_vals)

    def get_shot_measured_value(self, index: int) -> CircuitShotMeasuredValue:
        """Get measured value object of the shot specified by the index.

        Args:
            index (int): Index of shot

        Raises:
            ValueError: If `index` is invalid

        Returns:
            CircuitShotMeasuredValue: Measured value object of shot

        Example:
            >>> from mqc3.circuit import CircuitResult
            >>> from mqc3.circuit.result import CircuitShotMeasuredValue, CircuitOperationMeasuredValue
            >>> result = CircuitResult([
            ...     CircuitShotMeasuredValue([
            ...         CircuitOperationMeasuredValue(index=0, value=1.0),
            ...         CircuitOperationMeasuredValue(index=1, value=0.0),
            ...     ]),
            ...     CircuitShotMeasuredValue([
            ...         CircuitOperationMeasuredValue(index=0, value=0.0),
            ...         CircuitOperationMeasuredValue(index=1, value=1.0),
            ...     ]),
            ... ])
            >>> result.get_shot_measured_value(0)
            [CircuitOperationMeasuredValue(index=0, value=1.0), CircuitOperationMeasuredValue(index=1, value=0.0)]
        """
        if index < 0 or index >= len(self.measured_vals):
            message = "Index is invalid."
            raise ValueError(message)
        return self.measured_vals[index]

    def proto(self) -> PbCircuitResult:  # noqa: D102
        return PbCircuitResult(
            measured_vals=[
                PbCircuitResult.ShotMeasuredValue(
                    measured_vals=[
                        PbCircuitResult.OperationMeasuredValue(index=omv.index, value=omv.value) for omv in smv
                    ],
                )
                for smv in self.measured_vals
            ],
        )

    @staticmethod
    def construct_from_proto(proto_result: PbCircuitResult) -> CircuitResult:  # noqa: D102
        return CircuitResult([
            CircuitShotMeasuredValue(
                CircuitOperationMeasuredValue(index=omv.index, value=omv.value) for omv in smv.measured_vals
            )
            for smv in proto_result.measured_vals
        ])

    def save(self, path: str | Path, proto_format: ProtoFormat = "text") -> None:  # noqa: D102
        save(self.proto(), path, proto_format)

    @staticmethod
    def load(path: str | Path, proto_format: ProtoFormat = "text") -> CircuitResult:  # noqa: D102
        return CircuitResult.construct_from_proto(load(PbCircuitResult, path, proto_format))
