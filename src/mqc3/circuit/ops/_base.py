"""Base instruction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from mqc3.feedforward import FeedForward, Variable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation


@dataclass
class QuMode:
    """Quantum mode."""

    id: int
    "Index of the mode."

    def __str__(self) -> str:
        """Return the index of the mode.

        Returns:
            str: Index of the mode.
        """
        return str(self.id)


@dataclass
class Operand:
    """Operand for optical quantum operations."""

    modes: list[QuMode]
    "List of modes."

    _program: "CircuitRepr"  # pyright: ignore[reportUndefinedVariable] # noqa: F821, UP037

    def get_ids(self) -> list[int]:
        """Get id list of modes.

        Returns:
            list[int]: List of mode ids.

        Examples:
            >>> from mqc3.circuit import Operand, QuMode
            >>> operand = Operand([QuMode(0), QuMode(1)], None)
            >>> operand.get_ids()
            [0, 1]
        """
        return [mode.id for mode in self.modes]

    def __len__(self) -> int:
        """Return the number of modes.

        Returns:
            int: The number of modes.
        """
        return len(self.modes)

    def __iter__(self) -> Iterator[QuMode]:
        """Return the iterator of the `modes` attribute.

        Yields:
            Iterator[QuMode]: Iterator of the `modes` attribute.
        """
        yield from self.modes

    def __str__(self) -> str:
        """Return the ID list of the modes.

        Returns:
            str: ID list.
        """
        return str(self.modes)

    def __repr__(self) -> str:
        """Return the ID list of the modes.

        Returns:
            str: ID list.
        """
        return str(self.modes)


class MeasuredVariable(Variable):
    """Measured variable for feedforward."""

    def __init__(self, operation: Operation) -> None:
        """Construct a measured variable.

        Args:
            operation (Operation): Operation.
        """
        self._operation = operation

    def get_from_operation(self) -> Operation:
        """Get the operation that the variable is from.

        Returns:
            Operation: Operation.
        """
        return self._operation


CircOpParam: TypeAlias = FeedForward[MeasuredVariable] | float  # noqa: UP040


class Operation(ABC):
    """Operations performed in an optical quantum circuit.

    Specific operations are implemented by inheriting from this abstract class.
    See the :mod:`~mqc3.circuit.ops.intrinsic` and :mod:`~mqc3.circuit.ops.std`
    modules for concrete implementations.
    """

    def __init__(self) -> None:
        """Construct an operation.

        Examples:
            >>> from mqc3.circuit.ops import intrinsic
            >>> op = intrinsic.Measurement(0.0)
        """
        self._opnd: Operand | None = None

    def __ror__(self, opnd: Operand) -> Operand:
        """Apply the operation to the input mode.

        Args:
            opnd (Operand): The mode to apply the operation.

        Returns:
            Operand: The mode to which the operation has been applied.
        """
        self._check_n_modes(opnd)
        self._opnd = opnd
        program = opnd._program
        program._insert(self)

        return opnd

    def _check_n_modes(self, opnd: Operand) -> None:
        """Check the number of modes in an operand.

        Raises:
            RuntimeError: If the check fails.
        """
        expected = self.n_modes()
        if len(opnd) != expected:
            message = f"The number of modes for operation `{self.name()}` must be {expected}."
            raise RuntimeError(message)

    def opnd(self) -> Operand:
        """Get the operand of the operation.

        Returns:
            Operand: Operand of the operation.

        Raises:
            RuntimeError: If the operand is not set.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> circuit = CircuitRepr("example")
            >>> op = intrinsic.PhaseRotation(0.0)
            >>> circuit.Q(0) | op
            [QuMode(id=0)]
            >>> op.opnd().get_ids()
            [0]
        """
        if self._opnd is None:
            message = "Operand is not set."
            raise RuntimeError(message)
        return self._opnd

    def __str__(self) -> str:
        """Return a string representation of the operation.

        Returns:
            str: String representation of the operation.
        """
        if self._opnd is None:
            return f"[{self.name()}] {self.parameters()}"
        return f"[{self.name()}] {self.parameters()} {self._opnd}"

    @abstractmethod
    def name(self) -> str:
        """Get the name of the operation.

        Returns:
            str: The name of operation.

        Examples:
            >>> from mqc3.circuit.ops import intrinsic
            >>> op = intrinsic.Measurement(0.0)
            >>> op.name()
            'intrinsic.measurement'
        """

    @abstractmethod
    def n_modes(self) -> int:
        """Get the number of modes.

        Returns:
            int: The number of target modes.

        Examples:
            >>> from mqc3.circuit.ops import intrinsic
            >>> op = intrinsic.Measurement(0.0)
            >>> op.n_modes()
            1
        """

    @abstractmethod
    def n_macronodes(self) -> int:
        """Get the number of macronodes to run this operation in the graph representation."""

    @abstractmethod
    def parameters(self) -> list[CircOpParam]:
        """Get the parameters of the operation.

        Returns:
            list[CircOpParam]: Parameters.
        """

    @abstractmethod
    def to_intrinsic_ops(self) -> list[Operation]:
        """Convert to an equivalent sequence of intrinsic operations.

        Returns:
            list[Operation]: Intrinsic operations.
        """

    def has_feedforward_param(self) -> bool:
        """Check if the operation has any feedforward parameter.

        Returns:
            bool: Whether the operation has any feedforward parameter.
        """
        return any(isinstance(param, FeedForward) for param in self.parameters())

    @abstractmethod
    def proto(self) -> list[PbOperation]:
        """Get the proto of the operation."""
