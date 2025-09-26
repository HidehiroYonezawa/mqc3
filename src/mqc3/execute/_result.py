from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsIndex

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from datetime import timedelta

    from mqc3.client.abstract import AbstractClientResult, MeasuredValue, ReprType, ResultType


@dataclass(frozen=True)
class ExecutionResult:
    """The result of executing a quantum circuit with a client."""

    total_time: timedelta
    """Total time taken to execute the quantum circuit.

    ``total_time`` is the time between the start and the end of :func:`~mqc3.execute.execute`.
    """

    input_repr: ReprType
    """Input representation."""

    execution_result: ResultType
    """Execution result."""

    client_result: AbstractClientResult
    """The raw result returned from the client backend."""

    n_shots: int
    """The number of shots."""

    @property
    def execution_time(self) -> timedelta:
        """Get the execution time.

        Returns:
            timedelta: Execution time.
        """
        return self.client_result.execution_time

    def __len__(self) -> int:
        """Return the number of shots.

        Returns:
            int: The number of shots.
        """
        return self.n_shots

    def __iter__(self) -> Iterator:
        """Iterator of the result.

        The iterator is the same as that of ``self.execution_result``.

        Yields:
            Iterator: Iterator of the result
        """
        yield from self.execution_result

    def get_shot_measured_value(self, index: int) -> MeasuredValue:
        """Get the measured value of the shot at the index.

        This function gets values from ``self.execution_result``.

        Args:
            index (int): Shot index.

        Returns:
            MeasuredValue: Measured value.
        """
        return self.execution_result.get_shot_measured_value(index)

    def __getitem__(self, index: int | slice | SupportsIndex) -> MeasuredValue | Sequence[MeasuredValue]:
        """Get the measured value of the shot at the index.

        This function gets values from ``self.execution_result``.

        Args:
            index (int | slice | SupportsIndex): Index or slice.

        Returns:
            MeasuredValue | Sequence[MeasuredValue]: Measured value or sequence of measured values.
        """
        return self.execution_result.measured_vals[index]
