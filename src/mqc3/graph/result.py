"""Computation result in graph representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsIndex

from mqc3.pb.io import ProtoFormat, load, save
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphResult as PbGraphResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from mqc3.machinery.program import MachineryRepr
    from mqc3.machinery.result import MachineryResult


@dataclass(frozen=True)
class GraphMacronodeMeasuredValue:
    """Measured values in a macronode specified by the index."""

    index: int
    "Index of the measured macronode."

    h: int
    "h-coordinate of the macronode."

    w: int
    "w-coordinate of the macronode."

    m_b: float
    "Measured value of micronode b."

    m_d: float
    "Measured value of micronode d."


class GraphShotMeasuredValue:
    """Overridden list class having GraphMacronodeMeasuredValue as its elements."""

    def __init__(self, items: Iterable[GraphMacronodeMeasuredValue], *, n_local_macronodes: int) -> None:
        """Constructor."""
        self.items: dict[int, GraphMacronodeMeasuredValue] = dict(
            sorted({mmv.index: mmv for mmv in items}.items()),
        )
        self.n_local_macronodes = n_local_macronodes

    def __getitem__(self, key: int | tuple[int, int]) -> GraphMacronodeMeasuredValue:
        """Get GraphMacronodeMeasuredValue object specifying index or coordinate.

        Args:
            key (int | tuple[int, int]): Index or coordinate of the macronode.

        Raises:
            TypeError: Specified key has wrong type.

        Returns:
            GraphMacronodeMeasuredValue: Specified object.
        """
        if isinstance(key, int):
            return self.items[key]
        if isinstance(key, tuple) and len(key) == 2:  # noqa: PLR2004
            h, w = key
            return self.items[h + w * self.n_local_macronodes]
        msg = "Invalid key type."
        raise TypeError(msg)

    def __len__(self) -> int:
        """The number of macronodes from which the measured values were read out.

        Returns:
            int: The number of macronodes from which the measured values were read out.
        """
        return len(self.items)

    def __iter__(self) -> Iterator:
        """Iterator of items.

        Yields:
            Iterator: Iterator of items.
        """
        yield from self.items.values()

    def __str__(self) -> str:
        """String representation.

        Returns:
            str: String representation.
        """
        return f"{self.items}"

    def __repr__(self) -> str:
        """String representation.

        Returns:
            str: String representation.
        """
        return f"{self.items}"

    def index_list(self) -> list[int]:
        """Get index list of the macronodes sorted in ascending order.

        Returns:
            list[int]: Index list of the macronodes sorted in ascending order.
        """
        return list(self.items.keys())

    def coord_list(self) -> list[tuple[int, int]]:
        """Get coordinate list of the macronodes sorted in ascending order.

        Returns:
            list[tuple[int, int]]: Coordinate list of the macronodes sorted in ascending order.
        """
        return [(idx % self.n_local_macronodes, idx // self.n_local_macronodes) for idx in self.items]


class GraphResult:
    """The measurement result in graph representation."""

    def __init__(self, n_local_macronodes: int, shot_measured_values: Iterable[GraphShotMeasuredValue]) -> None:
        """Constructor.

        Args:
            n_local_macronodes (int): Number of local macronodes in graph.
            shot_measured_values (Iterable[GraphShotMeasuredValue]): Iterable of GraphShotMeasuredValue objects.

        Raises:
            TypeError: If the input argument is not Iterable[GraphShotMeasuredValue].
        """
        if not all(isinstance(x, GraphShotMeasuredValue) for x in shot_measured_values):
            message = "The argument type must be Iterable[GraphShotMeasuredValue]."
            raise TypeError(message)
        self.n_local_macronodes = n_local_macronodes
        self.measured_vals: list[GraphShotMeasuredValue] = list(shot_measured_values)

    def __iter__(self) -> Iterator:
        """Iterator of `measured_vals` attribute.

        Yields:
            Iterator: Iterator of `measured_vals` attribute.

        Example:
            >>> from mqc3.graph import GraphResult
            >>> from mqc3.graph.result import GraphShotMeasuredValue, GraphMacronodeMeasuredValue
            >>> result = GraphResult(
            ...     2,
            ...     [
            ...         GraphShotMeasuredValue(
            ...             [
            ...                 GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=1.0, m_d=0.0),
            ...                 GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=0.0, m_d=1.0),
            ...             ],
            ...             n_local_macronodes=2,
            ...         ),
            ...         GraphShotMeasuredValue(
            ...             [
            ...                 GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=0.0, m_d=1.0),
            ...                 GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=1.0, m_d=0.0),
            ...             ],
            ...             n_local_macronodes=2,
            ...         ),
            ...     ],
            ... )
            >>> for shot_measured_value in result:
            ...     for macronode_measured_value in shot_measured_value:
            ...         print(macronode_measured_value)
            GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=1.0, m_d=0.0)
            GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=0.0, m_d=1.0)
            GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=0.0, m_d=1.0)
            GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=1.0, m_d=0.0)
        """
        yield from self.measured_vals

    def __getitem__(
        self,
        index: int | slice | SupportsIndex,
    ) -> GraphShotMeasuredValue | Sequence[GraphShotMeasuredValue]:
        """Get the measured value of the shot at the index.

        Args:
            index (int | slice | SupportsIndex): Index or slice.

        Returns:
            GraphShotMeasuredValue | Sequence[GraphShotMeasuredValue]: Measured value or sequence of measured values.
        """
        return self.measured_vals[index]

    def __delitem__(self, index: int | slice) -> None:
        """Delete items in `measured_vals`.

        Args:
            index (int | slice): Indices of `measured_vals` elements to delete.
        """
        del self.measured_vals[index]

    def __len__(self) -> int:
        """The number of shots.

        Returns:
            int: The number of shots.
        """
        return len(self.measured_vals)

    @staticmethod
    def construct_from_machinery_result(
        machinery_result: MachineryResult,
        machinery_input: MachineryRepr,
    ) -> GraphResult:
        """Construct and return the result object from the machinery representation.

        Information from the input in machinery representation is also used.

        Args:
            machinery_result (MachineryResult): Result object in machinery representation.
            machinery_input (MachineryRepr): Input in machinery representation.

        Returns:
            GraphResult: The result object.
        """
        return GraphResult(
            machinery_input.n_local_macronodes,
            [
                GraphShotMeasuredValue(
                    [
                        GraphMacronodeMeasuredValue(
                            index=mmv.index,
                            h=mmv.index % machinery_input.n_local_macronodes,
                            w=mmv.index // machinery_input.n_local_macronodes,
                            m_b=mmv.m_b,
                            m_d=mmv.m_d,
                        )
                        for mmv in smv
                    ],
                    n_local_macronodes=machinery_input.n_local_macronodes,
                )
                for smv in machinery_result.measured_vals
            ],
        )

    def n_shots(self) -> int:
        """Get number of shots in `measured_vals` attribute.

        Returns:
            int: The number of shots in `measured_vals` attribute.

        Example:
            >>> from mqc3.graph import GraphResult
            >>> from mqc3.graph.result import GraphShotMeasuredValue, GraphMacronodeMeasuredValue
            >>> result = GraphResult(
            ...     2,
            ...     [
            ...         GraphShotMeasuredValue(
            ...             [
            ...                 GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=1.0, m_d=0.0),
            ...                 GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=0.0, m_d=1.0),
            ...             ],
            ...             n_local_macronodes=2,
            ...         ),
            ...         GraphShotMeasuredValue(
            ...             [
            ...                 GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=0.0, m_d=1.0),
            ...                 GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=1.0, m_d=0.0),
            ...             ],
            ...             n_local_macronodes=2,
            ...         ),
            ...     ],
            ... )
            >>> result.n_shots()
            2
        """
        return len(self.measured_vals)

    def get_shot_measured_value(self, index: int) -> GraphShotMeasuredValue:
        """Get measured value object of the shot specified by the index.

        Args:
            index (int): Index of shot.

        Returns:
            GraphShotMeasuredValue: Measured value object of shot.

        Raises:
            ValueError: If `index` is invalid.

        Example:
            >>> from mqc3.graph import GraphResult
            >>> from mqc3.graph.result import GraphShotMeasuredValue, GraphMacronodeMeasuredValue
            >>> result = GraphResult(
            ...     2,
            ...     [
            ...         GraphShotMeasuredValue(
            ...             [
            ...                 GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=1.0, m_d=0.0),
            ...                 GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=0.0, m_d=1.0),
            ...             ],
            ...             n_local_macronodes=2,
            ...         ),
            ...         GraphShotMeasuredValue(
            ...             [
            ...                 GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=0.0, m_d=1.0),
            ...                 GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=1.0, m_d=0.0),
            ...             ],
            ...             n_local_macronodes=2,
            ...         ),
            ...     ],
            ... )
            >>> result.get_shot_measured_value(0)
            {0: GraphMacronodeMeasuredValue(index=0, h=0, w=0, m_b=1.0, m_d=0.0), 1: GraphMacronodeMeasuredValue(index=1, h=1, w=0, m_b=0.0, m_d=1.0)}
        """  # noqa: E501
        if index < 0 or index >= len(self.measured_vals):
            message = "Index is invalid."
            raise ValueError(message)
        return self.measured_vals[index]

    def append(self, other: GraphResult) -> None:
        """Append the other result to this result.

        Args:
            other (GraphResult): Append target.
        """
        self.measured_vals.extend(other.measured_vals)

    def proto(self) -> PbGraphResult:  # noqa: D102
        return PbGraphResult(
            n_local_macronodes=self.n_local_macronodes,
            measured_vals=[
                PbGraphResult.ShotMeasuredValue(
                    measured_vals=[
                        PbGraphResult.MacronodeMeasuredValue(index=mmv.index, m_b=mmv.m_b, m_d=mmv.m_d) for mmv in smv
                    ],
                )
                for smv in self.measured_vals
            ],
        )

    @staticmethod
    def construct_from_proto(proto_result: PbGraphResult) -> GraphResult:  # noqa: D102
        return GraphResult(
            proto_result.n_local_macronodes,
            [
                GraphShotMeasuredValue(
                    [
                        GraphMacronodeMeasuredValue(
                            index=mmv.index,
                            h=mmv.index % proto_result.n_local_macronodes,
                            w=mmv.index // proto_result.n_local_macronodes,
                            m_b=mmv.m_b,
                            m_d=mmv.m_d,
                        )
                        for mmv in smv.measured_vals
                    ],
                    n_local_macronodes=proto_result.n_local_macronodes,
                )
                for smv in proto_result.measured_vals
            ],
        )

    def save(self, path: str | Path, proto_format: ProtoFormat = "text") -> None:  # noqa: D102
        save(self.proto(), path, proto_format)

    @staticmethod
    def load(path: str | Path, proto_format: ProtoFormat = "text") -> GraphResult:  # noqa: D102
        return GraphResult.construct_from_proto(load(PbGraphResult, path, proto_format))
