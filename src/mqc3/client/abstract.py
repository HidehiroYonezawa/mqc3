"""Abstract client class."""

from abc import ABC, abstractmethod
from datetime import timedelta

from mqc3.circuit import CircuitRepr, CircuitResult
from mqc3.circuit.result import CircuitShotMeasuredValue
from mqc3.graph import GraphRepr, GraphResult
from mqc3.graph.result import GraphShotMeasuredValue
from mqc3.machinery import MachineryRepr, MachineryResult
from mqc3.machinery.result import MachineryShotMeasuredValue

ReprType = CircuitRepr | GraphRepr | MachineryRepr
ResultType = CircuitResult | GraphResult | MachineryResult
MeasuredValue = CircuitShotMeasuredValue | GraphShotMeasuredValue | MachineryShotMeasuredValue


class AbstractClientResult(ABC):
    """Abstract client result class."""

    execution_time: timedelta
    """Elapsed time to execute a quantum circuit."""

    @property
    @abstractmethod
    def execution_result(self) -> ResultType:
        """Raw execution result from the client.

        Returns:
            ResultType: Execution result.
        """
        raise NotImplementedError


class AbstractClient(ABC):
    """Abstract client class."""

    n_shots: int

    def __init__(self, n_shots: int) -> None:
        """Abstract client class constructor.

        Args:
            n_shots (int): The number of shots.
        """
        self.n_shots = n_shots

    @abstractmethod
    def run(self, representation: ReprType) -> AbstractClientResult:
        """Run a representation.

        Note:
            This method is synchronous in the sense that it submits a job and waits for the result.

        Args:
            representation (ReprType): Representation which is suitable for the client.

        Returns:
            ResultType: Result of the running.
        """
        raise NotImplementedError
