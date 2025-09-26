from datetime import datetime
from zoneinfo import ZoneInfo

from mqc3.client.abstract import AbstractClient, ReprType
from mqc3.execute._result import ExecutionResult


def execute(representation: ReprType, client: AbstractClient) -> ExecutionResult:
    """Execute a circuit, graph or machinery representation.

    Args:
        representation (AbstractReprType): The representation to be executed.
        client (AbstractClient): The execution backend client.

    Returns:
        ExecutionResult: The result of the execution.

    Examples:
        Using :class:`~mqc3.client.MQC3Client`:

        >>> from mqc3.circuit import CircuitRepr
        >>> from mqc3.circuit.ops import intrinsic
        >>> from mqc3.client import MQC3Client
        >>> from mqc3.execute import execute
        >>> circuit = CircuitRepr("example")
        >>> circuit.Q(0) | intrinsic.PhaseRotation(1)
        [QuMode(id=0)]
        >>> circuit.Q(1) | intrinsic.PhaseRotation(2)
        [QuMode(id=1)]
        >>> circuit.Q(0, 1) | intrinsic.ControlledZ(3)
        [QuMode(id=0), QuMode(id=1)]
        >>> circuit.Q(0) | intrinsic.Measurement(0.0)
        Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
        >>> circuit.Q(1) | intrinsic.Measurement(0.0)
        Var([intrinsic.measurement] [0.0] [QuMode(id=1)])
        >>> client = MQC3Client(n_shots=1, backend="emulator")
        >>> result = execute(circuit, client)

        Using :class:`~mqc3.client.SimulatorClient`:

        >>> from mqc3.circuit.state import BosonicState
        >>> from mqc3.client import SimulatorClient
        >>> circuit = CircuitRepr("example")
        >>> circuit.Q(0) | intrinsic.PhaseRotation(1)
        [QuMode(id=0)]
        >>> circuit.Q(1) | intrinsic.PhaseRotation(2)
        [QuMode(id=1)]
        >>> circuit.Q(0, 1) | intrinsic.ControlledZ(3)
        [QuMode(id=0), QuMode(id=1)]
        >>> circuit.Q(0) | intrinsic.Measurement(0.0)
        Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
        >>> circuit.Q(1) | intrinsic.Measurement(0.0)
        Var([intrinsic.measurement] [0.0] [QuMode(id=1)])
        >>> circuit.set_initial_state(0, BosonicState.vacuum())
        >>> circuit.set_initial_state(1, BosonicState.vacuum())
        >>> client = SimulatorClient(n_shots=1, remote=False, state_save_policy="all")
        >>> result = execute(circuit, client)
    """
    started_at = datetime.now(ZoneInfo("Asia/Tokyo"))

    client_result = client.run(representation)

    finished_at = datetime.now(started_at.tzinfo)
    total_time = finished_at - started_at
    return ExecutionResult(
        input_repr=representation,
        client_result=client_result,
        total_time=total_time,
        execution_result=client_result.execution_result,
        n_shots=client.n_shots,
    )
