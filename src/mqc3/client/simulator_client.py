"""Simulator client for optical quantum computing."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal, SupportsIndex
from zoneinfo import ZoneInfo

import requests
from google.protobuf import duration_pb2
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from mqc3.__version__ import __version__
from mqc3.circuit import CircuitRepr, CircuitResult
from mqc3.circuit.state import BosonicState
from mqc3.client._grpc_core import GrpcConnectionManager
from mqc3.client._safe_save import safe_save
from mqc3.client.abstract import AbstractClient, AbstractClientResult, MeasuredValue, ReprType, ResultType
from mqc3.graph import GraphRepr, GraphResult
from mqc3.machinery import MachineryResult
from mqc3.pb.mqc3_cloud.program.v1 import quantum_program_pb2
from mqc3.pb.mqc3_cloud.scheduler.v1 import job_pb2, submission_pb2, submission_pb2_grpc

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from zoneinfo import ZoneInfo


# gRPC settings
GRPC_CLIENT_MAX_RETRIES = 3
MAX_SUBMISSION_CHECK = 3
MAX_MESSAGE_LENGTH = int(os.getenv("GRPC_MAX_MESSAGE_LENGTH", str(1024 * 1024 * 1024)))

_TZ_UTC = timezone.utc  # noqa: UP017


@dataclass
class JobTimeline:
    """Timeline of a job execution in UTC."""

    submitted_at: datetime | None = None
    queued_at: datetime | None = None
    dequeued_at: datetime | None = None
    execution_started_at: datetime | None = None
    execution_finished_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def from_proto(cls, timestamps: job_pb2.JobTimestamps, timezone: timezone | ZoneInfo = _TZ_UTC) -> JobTimeline:
        """Construct from a JobTimestamps proto.

        Args:
            timestamps (job_pb2.JobTimestamps): JobTimestamps type of proto.
            timezone (timezone | ZoneInfo, optional): Timezone to use (default: UTC).

        Returns:
            JobTimeline: JobTimeline object.
        """
        return cls(
            submitted_at=timestamps.submitted_at.ToDatetime(timezone) if timestamps.HasField("submitted_at") else None,
            queued_at=timestamps.queued_at.ToDatetime(timezone) if timestamps.HasField("queued_at") else None,
            dequeued_at=timestamps.dequeued_at.ToDatetime(timezone) if timestamps.HasField("dequeued_at") else None,
            execution_started_at=timestamps.execution_started_at.ToDatetime(timezone)
            if timestamps.HasField("execution_started_at")
            else None,
            execution_finished_at=timestamps.execution_finished_at.ToDatetime(timezone)
            if timestamps.HasField("execution_finished_at")
            else None,
            finished_at=timestamps.finished_at.ToDatetime(timezone) if timestamps.HasField("finished_at") else None,
        )

    @property
    def wait_time(self) -> timedelta | None:
        """Return the waiting time of the job.

        ``wait_time`` is the time between the job being queued and the job being dequeued.

        Returns:
            timedelta | None: Waiting time of the job.
        """
        if self.queued_at is None or self.dequeued_at is None:
            return None
        return self.dequeued_at - self.queued_at

    @property
    def execution_time(self) -> timedelta | None:
        """Return the execution time of the job.

        Returns:
            timedelta | None: Execution time of the job.
        """
        if self.execution_started_at is None or self.execution_finished_at is None:
            return None
        return self.execution_finished_at - self.execution_started_at

    @property
    def total_time(self) -> timedelta | None:
        """Return the total time of the job.

        ``total_time`` is the time between the job being submitted and the job being finished.

        Returns:
            timedelta | None: Total time of the job.
        """
        if self.submitted_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.submitted_at


@dataclass
class ExecutionDetails:
    """Versions and timestamps related to a job execution."""

    # Versions
    scheduler_version: str
    simulator_version: str

    # Timeline
    timeline: JobTimeline | None = None

    @classmethod
    def from_proto(
        cls, execution_details: job_pb2.JobExecutionDetails, timezone: timezone | ZoneInfo = _TZ_UTC
    ) -> ExecutionDetails:
        """Construct from a JobExecutionDetails proto.

        Args:
            execution_details (job_pb2.JobExecutionDetails): ExecutionDetails type of proto.
            timezone (timezone | ZoneInfo, optional): Timezone to use (default: UTC).

        Returns:
            ExecutionDetails: ExecutionDetails object.
        """
        timeline = None
        if execution_details.HasField("timestamps"):
            timeline = JobTimeline.from_proto(execution_details.timestamps, timezone)

        return cls(
            scheduler_version=execution_details.version.scheduler_version,
            simulator_version=execution_details.version.simulator_version,
            timeline=timeline,
        )


def _construct_result_from_proto(
    proto_result: quantum_program_pb2.QuantumProgramResult,
    proto_execution_details: job_pb2.JobExecutionDetails,
    timezone: timezone | ZoneInfo = _TZ_UTC,
) -> SimulatorClientResult:
    """Construct an ExecutionResult from a proto.

    Args:
        proto_result (quantum_program_pb2.QuantumProgramResult): Proto of the result.
        proto_execution_details (job_pb2.JobExecutionDetails): Proto of the execution details.
        representation (ReprType): Representation.
        timezone (timezone | ZoneInfo, optional): Timezone to use (default: UTC).

    Returns:
        SimulatorClientResult: Result.

    Raises:
        ValueError: If proto_result is empty.
        RuntimeError: If failed to construct states after simulation from proto.
    """
    if not proto_result:
        msg = "Empty proto list."
        raise ValueError(msg)

    circuit_result = None
    graph_result = None
    machinery_result = None

    if proto_result.HasField("circuit_result"):
        circuit_result = CircuitResult.construct_from_proto(proto_result.circuit_result)
    if proto_result.HasField("graph_result"):
        graph_result = GraphResult.construct_from_proto(proto_result.graph_result)
    if proto_result.HasField("machinery_result"):
        machinery_result = MachineryResult.construct_from_proto(proto_result.machinery_result)

    states = []
    for i, state in enumerate(proto_result.circuit_state):
        tmp = BosonicState.construct_from_proto(state)
        if tmp is None:
            msg = f"Failed to construct state at shot '{i}'. SDK update may resolve this problem."
            raise RuntimeError(msg)
        states.append(tmp)

    return SimulatorClientResult(
        execution_details=ExecutionDetails.from_proto(proto_execution_details, timezone),
        _circuit_result=circuit_result,
        _graph_result=graph_result,
        _machinery_result=machinery_result,
        states=states,
    )


@dataclass(frozen=True)
class SimulatorClientResult(AbstractClientResult):
    """The result of executing a representation with a SimulatorClient."""

    execution_details: ExecutionDetails
    """Execution details includes version of backend and the timeline of the job."""

    _circuit_result: CircuitResult | None
    """Measurement results after circuit execution"""

    _graph_result: GraphResult | None
    """Measurement results after graph execution"""

    _machinery_result: MachineryResult | None
    """Measurement results after machinery execution"""

    states: list[BosonicState]
    """States after circuit execution."""

    @property
    def circuit_result(self) -> CircuitResult:
        """Return the circuit execution result.

        Returns:
            CircuitResult: The circuit execution result.

        Raises:
            ValueError: If no circuit result is available. This usually means the
                circuit has not been executed yet or execution did not produce a
                circuit-level result.
        """
        if self._circuit_result is None:
            msg = (
                "Circuit result is not available. Run the circuit first or verify "
                "that execution produced a circuit result."
            )
            raise ValueError(msg)
        return self._circuit_result

    @property
    def graph_result(self) -> GraphResult:
        """Return the graph execution result.

        Returns:
            GraphResult: The graph execution result.

        Raises:
            ValueError: If no graph result is available. This usually means the
                graph-level execution has not been performed yet or it produced no
                result.
        """
        if self._graph_result is None:
            msg = (
                "Graph result is not available. Ensure that graph execution has been performed and produced a result."
            )
            raise ValueError(msg)
        return self._graph_result

    @property
    def machinery_result(self) -> MachineryResult:
        """Return the machinery execution result.

        Returns:
            MachineryResult: The machinery execution result.

        Raises:
            ValueError: If no machinery result is available. This usually means
                machinery execution has not been performed yet or it produced
                no result.
        """
        if self._machinery_result is None:
            msg = (
                "Machinery result is not available. Ensure that machinery "
                "execution has been performed and produced a result."
            )
            raise ValueError(msg)
        return self._machinery_result

    @property
    def execution_result(self) -> ResultType:
        """Get the result of the top layer in the execution.

        Returns:
            CircuitResult: Result.

        Examples:
            >>> from mqc3.client import SimulatorClient
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> from mqc3.circuit.state import BosonicState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0.1, 0.2)
            [QuMode(id=0)]
            >>> circuit.Q(0)| intrinsic.Measurement(0.0)
            Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
            >>> circuit.set_initial_state(0, BosonicState.vacuum())
            >>> client = SimulatorClient(n_shots=1, backend="cpu")
            >>> result = client.run(circuit)
            >>> circuit_result = result.execution_result

        Raises:
            ValueError: Raised when no execution result is available, i.e.,
                none of ``circuit_result``, ``graph_result``, or ``machinery_result``
                has been set (for example, if the circuit has not been run).
        """
        if self._circuit_result is not None:
            return self._circuit_result
        if self._graph_result is not None:
            return self._graph_result
        if self._machinery_result is not None:
            return self._machinery_result
        msg = ""
        raise ValueError(msg)

    @property
    def n_shots(self) -> int:
        """Return the number of shots.

        Returns:
            int: The number of shots.

        Examples:
            >>> from mqc3.client import SimulatorClient
            >>> from mqc3.execute import execute
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> from mqc3.circuit.state import BosonicState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0.1, 0.2)
            [QuMode(id=0)]
            >>> circuit.Q(0)| intrinsic.Measurement(0.0)
            Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
            >>> circuit.set_initial_state(0, BosonicState.vacuum())
            >>> client = SimulatorClient(n_shots=1, backend="cpu")
            >>> result = execute(circuit, client)
            >>> result.n_shots
            1
        """
        return self.execution_result.n_shots()

    @property
    def wait_time(self) -> timedelta | None:
        """Return the waiting time of the job.

        ``wait_time`` is the time between the job being queued and the job being dequeued.

        Returns:
            timedelta | None: The waiting time of the job.
        """
        if self.execution_details.timeline is None:
            return None

        return self.execution_details.timeline.wait_time

    @property
    def execution_time(self) -> timedelta | None:
        """Return the execution time of the job.

        Returns:
            timedelta | None: The execution time of the job.
        """
        if self.execution_details.timeline is None:
            return None

        return self.execution_details.timeline.execution_time

    @property
    def total_time(self) -> timedelta | None:
        """Return the total time of the job.

        ``total_time`` is the time between the job being submitted and the job being finished.

        Returns:
            timedelta | None: The total time of the job.
        """
        if self.execution_details.timeline is None:
            return None

        return self.execution_details.timeline.total_time

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
            Iterator: Iterator of the result.
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

    def __getitem__(
        self,
        index: int | slice | SupportsIndex,
    ) -> MeasuredValue | Sequence[MeasuredValue]:
        """Get the measured value of the shot at the index.

        This function gets values from ``self.execution_result``.

        Args:
            index (int | slice | SupportsIndex): Index or slice.

        Returns:
            MeasuredValue | Sequence[MeasuredValue]: Measured value or sequence of measured values.
        """
        return self.execution_result.measured_vals[index]


BackendType = Literal["cpu", "single_gpu", "multi_gpu"]
"""Backends for the simulator client.

This option is valid only when `remote` option in :class:`SimulatorClient` is ``True``.

- ``cpu``: Simulate circuits with CPU.
- ``single_gpu``: Simulate circuits with a single GPU.
- ``multi_gpu``: Simulate circuits with multi GPU.
"""

StateSavePolicy = Literal["all", "first_only", "none"]
"""Policy for saving states after simulation.

- ``all``: Save states of all shots.
- ``first_only``: Save only the state of the first shot.
- ``none``: Do not save any states..
"""


def _get_state_save_policy(state_save_policy: StateSavePolicy) -> job_pb2.JobStateSavePolicy:
    """Utility function to get the string of StateSavePolicy for gRPC.

    Args:
        state_save_policy (StateSavePolicy): Policy for saving states after simulation.

    Returns:
        str: StateSavePolicy for gRPC.
    """
    if state_save_policy == "all":
        return job_pb2.JobStateSavePolicy.JOB_STATE_SAVE_POLICY_ALL
    if state_save_policy == "first_only":
        return job_pb2.JobStateSavePolicy.JOB_STATE_SAVE_POLICY_FIRST_ONLY
    if state_save_policy == "none":
        return job_pb2.JobStateSavePolicy.JOB_STATE_SAVE_POLICY_NONE

    return job_pb2.JobStateSavePolicy.JOB_STATE_SAVE_POLICY_UNSPECIFIED


class SimulatorClient(AbstractClient, GrpcConnectionManager):
    """Simulator client for optical quantum computing."""

    token: str
    "MQC3 API token."

    backend: BackendType
    """Backend type of the simulator.

    This option is valid only when `remote` is ``True``.

    See :data:`BackendType`.
    """

    state_save_policy: StateSavePolicy
    """Policy for saving states after simulation.

    See :data:`StateSavePolicy`.
    """

    run_timeout: int
    """Timeout for running a job in seconds."""

    resource_squeezing_level: float
    """Squeezing level of the resource state. Used only for graph/machinery simulations."""

    read_timeout: float
    """Timeout for downloading a job result in seconds."""

    polling_interval: float
    """Polling interval in seconds."""

    timezone: timezone | ZoneInfo
    """Timezone to use."""

    submit_request_path: Path | None
    """Path to persist a copy of the `SubmitJobRequest` used for job submission.
    Set to `None` to disable saving.

    Notes:
        - The file may be overwritten by subsequent submissions.
          Use a unique per-job path (e.g., include a timestamp) if you need history.
    """

    result_response_path: Path | None
    """Path to persist a copy of the `GetJobResultResponse` metadata.
    This does not include the downloaded `QuantumProgramResult` payload.
    Set to `None` to disable saving.

    Notes:
        - The file may be overwritten; prefer a unique per-job path for history.
    """

    quantum_program_result_path: Path | None
    """Path to persist the `QuantumProgramResult` (the job's result payload).
    Set to `None` to disable saving.

    Notes:
        - The file may be overwritten; prefer a unique per-job path for history.
    """

    remote: bool = True
    """If ``True``, simulates representations on a remote server.

    If `remote` is ``False``, the following limitations apply:

    - Supports only circuit representations.
    - Circuits containing `intrinsic.Manual` cannot be executed.
    - All modes must have explicitly defined initial states.
    - Each mode's initial state must contain exactly one peak.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        n_shots: int = 1024,
        token: str = "",
        backend: BackendType = "cpu",
        state_save_policy: StateSavePolicy = "none",
        run_timeout: int = 180,
        resource_squeezing_level: float = 10.0,
        *,
        polling_interval: float = 1.0,
        connection_timeout: float = 60.0,
        read_timeout: float = 180.0,
        timezone: timezone | ZoneInfo = _TZ_UTC,
        max_send_message_length: int = MAX_MESSAGE_LENGTH,
        max_receive_message_length: int = MAX_MESSAGE_LENGTH,
        max_retry_count: int = GRPC_CLIENT_MAX_RETRIES,
        ca_cert_file: str | None = None,
        submit_request_path: Path | None = None,
        result_response_path: Path | None = None,
        quantum_program_result_path: Path | None = None,
        remote: bool = True,
    ) -> None:
        """SimulatorClient constructor.

        Args:
            n_shots (int): The number of shots.
            token (str): MQC3 API token.
            backend (:data:`BackendType`): Backend type (cpu, single_gpu or multi_gpu).
            state_save_policy (:data:`StateSavePolicy`): Policy for saving states after simulation
                (all, first_only or none).
            run_timeout (int): Timeout for executing a job in seconds.
            resource_squeezing_level (float): Squeezing level of the resource state.
                Used only for graph/machinery simulations.
            polling_interval (float): Polling interval in seconds (default: 1.0).
            connection_timeout (float): Timeout for establishing a connection when downloading a result in seconds.
            read_timeout (float): Timeout for downloading a result and gRPC connection in seconds.
            timezone (timezone | ZoneInfo): Timezone to use (default: UTC).
            max_send_message_length (int): Maximum size (in bytes) for outbound messages.
            max_receive_message_length (int): Maximum size (in bytes) for inbound messages.
            max_retry_count (int): Maximum number of retries for gRPC operations.
            ca_cert_file (str | None): Path to an SSL/TLS certificate file for secure communication.
                If None, the system's default trusted certificate store is used for verification.
                If you want to use insecure_channel, set MQC3_CLIENT_SECURE_CHANNEL environment variable to "false".
            submit_request_path (Path | None): If provided, writes the `SubmitJobRequest`
                used for submission. The file may be overwritten on subsequent submissions;
                use a unique per-job path (e.g., including `job_id` or a timestamp) to keep history.
            result_response_path (Path | None): If provided, writes the `GetJobResultResponse`.
                This does not include the downloaded result payload. Use a unique per-job path to avoid overwrites.
            quantum_program_result_path (Path | None): If provided, writes the `QuantumProgramResult`
                payload after a successful download. Use a unique per-job path to avoid overwrites.
            remote (bool): If true, simulate representations on remote server.

        Examples:
            >>> from mqc3.client import SimulatorClient
            >>> client = SimulatorClient(
            ...     n_shots=1024,
            ...     backend="cpu",
            ...     state_save_policy="first_only",
            ...     token="YOUR_API_TOKEN",
            ... )
        """
        self.token = os.getenv(key="MQC3_CLIENT_TOKEN", default=token)
        AbstractClient.__init__(self, n_shots=n_shots)
        GrpcConnectionManager.__init__(
            self,
            address=os.getenv(key="MQC3_SIMULATOR_CLIENT_URL", default="mqc3.com:8087"),
            max_send_message_length=max_send_message_length,
            max_receive_message_length=max_receive_message_length,
            max_retry_count=max_retry_count,
            timeout_sec=connection_timeout,
            secure_remote=os.getenv(key="MQC3_SIMULATOR_CLIENT_SECURE_CHANNEL", default="true").lower() == "true",
            ca_cert_file=ca_cert_file,
        )

        self.backend = backend
        self.state_save_policy = state_save_policy
        self.run_timeout = run_timeout
        self.resource_squeezing_level = resource_squeezing_level
        self.polling_interval = max(polling_interval, 1.0)
        self.read_timeout = read_timeout
        self.timezone = timezone
        self.submit_request_path = submit_request_path
        self.result_response_path = result_response_path
        self.quantum_program_result_path = quantum_program_result_path
        self.remote = remote

    @property
    def url(self) -> str:
        """Get the URL of the MQC3 server."""
        return self.address

    @url.setter
    def url(self, new_url: str) -> None:
        """Set the URL of the MQC3 server."""
        self.address = new_url
        self.reconnect()

    @property
    def connection_timeout(self) -> float:
        """Get the timeout for establishing a connection in seconds."""
        return self.timeout_sec

    @connection_timeout.setter
    def connection_timeout(self, connection_timeout: float) -> None:
        """Set the timeout for establishing a connection in seconds.

        Args:
            connection_timeout (float): Timeout for establishing a connection in seconds.
        """
        self.timeout_sec = connection_timeout

    def _recreate_stub(self) -> None:
        self.stub = submission_pb2_grpc.SubmissionServiceStub(self.channel)

    @staticmethod
    def _format_job_status_to_str(pb_job_status: job_pb2.JobStatus) -> str:
        """Format the job status to string.

        Args:
            pb_job_status (job_pb2.JobStatus): The job status with protobuf format.

        Returns:
            str: The job status with string format.
        """
        return job_pb2.JobStatus.Name(pb_job_status).replace("JOB_STATUS_", "")

    @staticmethod
    def _format_service_status_to_str(pb_service_status: submission_pb2.ServiceStatus) -> str:
        """Format the service status to string.

        Args:
            pb_service_status (submission_pb2.ServiceStatus): The service status with protobuf format.

        Returns:
            str: The service status with string format.
        """
        return submission_pb2.ServiceStatus.Name(pb_service_status).replace("SERVICE_STATUS_", "")

    def submit(self, representation: ReprType) -> str:
        """Submit a representation as a job.

        Args:
            representation (ReprType): Representation.

        Returns:
            str: Job ID.

        Raises:
            RuntimeError: If the client is not in remote mode, or if the job submission fails.

        Example:
            >>> from numpy import pi
            >>> from mqc3.client import SimulatorClient
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> from mqc3.circuit.state import BosonicState
            >>> client = SimulatorClient(n_shots=1, backend="cpu")
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.PhaseRotation(pi / 2)
            [QuMode(id=0)]
            >>> circuit.Q(0) | intrinsic.Measurement(0.0)
            Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
            >>> circuit.set_initial_state(0, BosonicState.vacuum())
            >>> job_id = client.submit(circuit)
        """
        if not self.remote:
            msg = "This method can only be executed when the client is in remote mode."
            raise RuntimeError(msg)

        if isinstance(representation, CircuitRepr):
            program = quantum_program_pb2.QuantumProgram(
                format=quantum_program_pb2.REPRESENTATION_FORMAT_CIRCUIT,
                circuit=representation.proto(),
            )
        elif isinstance(representation, GraphRepr):
            program = quantum_program_pb2.QuantumProgram(
                format=quantum_program_pb2.REPRESENTATION_FORMAT_GRAPH,
                graph=representation.proto(),
            )
        else:
            program = quantum_program_pb2.QuantumProgram(
                format=quantum_program_pb2.REPRESENTATION_FORMAT_MACHINERY,
                machinery=representation.proto(),
            )

        settings = job_pb2.JobExecutionSettings(
            backend=self.backend,
            n_shots=self.n_shots,
            timeout=duration_pb2.Duration(seconds=self.run_timeout),
            state_save_policy=_get_state_save_policy(self.state_save_policy),
            resource_squeezing_level=self.resource_squeezing_level,
            # role=,    # role is set by the server.
        )
        job = job_pb2.Job(program=program, settings=settings)
        request = submission_pb2.SubmitJobRequest(
            token=self.token,
            job=job,
            options=job_pb2.JobManagementOptions(save_job=False),
            sdk_version=__version__,
        )

        if self.submit_request_path:
            error = safe_save(request, self.submit_request_path, "text")
            if error:
                sys.stderr.write(
                    f"Failed to save request of 'SubmitJob' RPC (path={self.submit_request_path!s}): {error}\n"
                )

        response: submission_pb2.SubmitJobResponse = self._call_rpc(self.stub.SubmitJob, request)
        if response.error.code:
            msg = f"Failed to submit job: {response.error.description} ({response.error.code})."
            raise RuntimeError(msg)

        for attempt in range(1, MAX_SUBMISSION_CHECK + 1):
            if self._is_job_submitted(response.job_id):
                break
            if attempt == MAX_SUBMISSION_CHECK:
                msg = (
                    f"Submission not confirmed. The submit call may have succeeded, but the job is not yet visible. "
                    "Please wait a moment and retry 'SimulatorClient.get_job_status'. "
                    "If the job still cannot be found, report this as an issue. "
                    f"[job_id={response.job_id}]"
                )
                raise RuntimeError(msg)

            time.sleep(1 / 3)  # Sleep 1/3 seconds

        return response.job_id

    def _is_job_submitted(self, job_id: str) -> bool:
        request = submission_pb2.GetJobStatusRequest(token=self.token, job_id=job_id)
        response: submission_pb2.GetJobStatusResponse = self._call_rpc(self.stub.GetJobStatus, request)
        return not response.error.code.startswith("NOT_FOUND")

    def get_job_status(self, job_id: str) -> tuple[str, str, ExecutionDetails]:
        """Get the status of a job.

        Args:
            job_id (str): Job ID.

        Returns:
            tuple[str, str, ExecutionDetails]: A tuple of (job_status, status_detail, execution_details)
                representing the current job state.

        Raises:
            RuntimeError: If the client is not in remote mode, or if the job status retrieval fails.
        """
        if not self.remote:
            msg = "This method can only be executed when the client is in remote mode."
            raise RuntimeError(msg)

        request = submission_pb2.GetJobStatusRequest(token=self.token, job_id=job_id)
        response: submission_pb2.GetJobStatusResponse = self._call_rpc(self.stub.GetJobStatus, request)
        if response.error.code:
            msg = f"Failed to get job status: {response.error.description} ({response.error.code})."
            raise RuntimeError(msg)

        status_detail = response.status_detail
        execution_details = ExecutionDetails.from_proto(
            execution_details=response.execution_details,
            timezone=self.timezone,
        )
        return self._format_job_status_to_str(response.status), status_detail, execution_details

    def wait_for_completion(self, job_id: str) -> tuple[str, str, ExecutionDetails]:
        """Poll the job status until it reaches a terminal status.

        Args:
            job_id (str): Job ID.

        Returns:
            tuple[str, str, ExecutionDetails]: A tuple of (job_status, status_detail, execution_details)
                representing the terminal job state.

        Raises:
            RuntimeError: If the client is not in remote mode.
        """
        if not self.remote:
            msg = "This method can only be executed when the client is in remote mode."
            raise RuntimeError(msg)

        current_status, status_detail, execution_details = self.get_job_status(job_id)

        while current_status not in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"}:
            current_status, status_detail, execution_details = self.get_job_status(job_id)
            time.sleep(self.polling_interval)

        return current_status, status_detail, execution_details

    def get_job_result(self, job_id: str) -> SimulatorClientResult:  # noqa: C901
        """Get the result of a job.

        Args:
            job_id (str): Job ID.

        Returns:
            SimulatorClientResult: The result of the job.

        Raises:
            RuntimeError: If the client is not in remote mode or if the job result retrieval fails because of following
            reasons.
            - if the response from server contains an error.
            - if the job is not completed.
            - if fails to download the result.
        """
        if not self.remote:
            msg = "This method can only be executed when the client is in remote mode."
            raise RuntimeError(msg)

        request = submission_pb2.GetJobResultRequest(token=self.token, job_id=job_id)
        response: submission_pb2.GetJobResultResponse = self._call_rpc(self.stub.GetJobResult, request)

        if self.result_response_path:
            error = safe_save(response, self.result_response_path, "text")
            if error:
                sys.stderr.write(
                    f"Failed to save response of 'GetJobResult' RPC "
                    f"(job_id={job_id} path={self.result_response_path!s}): {error}\n"
                )

        if response.error.code:
            msg = f"Failed to get job result: {response.error.description} ({response.error.code})."
            raise RuntimeError(msg)

        if response.status != job_pb2.JobStatus.JOB_STATUS_COMPLETED:
            msg = f"Failed to get job result: job status is {self._format_job_status_to_str(response.status)}."
            if response.status_detail:
                msg += f" {response.status_detail}"
            raise RuntimeError(msg)

        try:
            headers = {"Accept-Encoding": "gzip, deflated"}
            session = requests.Session()
            retry = Retry(
                total=5,
                connect=5,
                read=5,
                status=5,
                backoff_factor=0.5,
                status_forcelist=(408, 429, 500, 502, 503, 504),
                allowed_methods=["GET", "HEAD"],
                respect_retry_after_header=True,
            )
            adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            download_response = session.get(
                response.result.result_url,
                headers=headers,
                timeout=(self.connection_timeout, self.read_timeout),
            )
            download_response.raise_for_status()
        except requests.exceptions.ConnectTimeout as err:
            msg = (
                "Failed to get job result: connection timeout. "
                "Please wait a moment and retry 'MQC3Client.get_job_result'. "
                "Increasing the 'MQC3Client.connect_timeout' value can be effective. "
                "If the result still cannot be retrieved, report this as an issue. "
                f"[job_id={job_id}]."
            )
            raise RuntimeError(msg) from err
        except requests.exceptions.ReadTimeout as err:
            msg = (
                "Failed to get job result: read timeout. "
                "Please wait a moment and retry 'MQC3Client.get_job_result'. "
                "Increasing the 'MQC3Client.read_timeout' value can be effective. "
                "If the result still cannot be retrieved, report this as an issue. "
                f"[job_id={job_id}]."
            )
            raise RuntimeError(msg) from err
        except Exception as err:
            msg = f"Failed to get job result: {err}."
            raise RuntimeError(msg) from err

        if not download_response.ok:
            msg = f"Failed to get job result: {download_response.reason} ({download_response.status_code})."
            raise RuntimeError(msg)

        result = quantum_program_pb2.QuantumProgramResult()
        result.ParseFromString(download_response.content)
        del download_response

        if self.quantum_program_result_path:
            error = safe_save(result, self.quantum_program_result_path, "text")
            if error:
                sys.stderr.write(
                    f"Failed to save QuantumProgramResult blob "
                    f"(job_id={job_id} path={self.quantum_program_result_path!s}): {error}\n"
                )

        return _construct_result_from_proto(
            proto_result=result,
            proto_execution_details=response.execution_details,
            timezone=self.timezone,
        )

    def cancel_job(self, job_id: str) -> None:
        """Cancel a job execution.

        A job can only be canceled if its state is QUEUING.

        Args:
            job_id (str): Job ID.

        Raises:
            RuntimeError: If the client is not in remote mode or if the job cancellation fails.
        """
        if not self.remote:
            msg = "This method can only be executed when the client is in remote mode."
            raise RuntimeError(msg)

        request = submission_pb2.CancelJobRequest(token=self.token, job_id=job_id)
        response: submission_pb2.CancelJobResponse = self._call_rpc(self.stub.CancelJob, request)

        if response.error.code:
            msg = f"Failed to cancel the job: {response.error.description} ({response.error.code})."
            raise RuntimeError(msg)

    def get_service_availability(self) -> tuple[str, str]:
        """Get the availability of the service.

        Returns:
            tuple[str, str]: A tuple of (service_status, description) representing the current service state.

        Raises:
            RuntimeError: If the client is not in remote mode or if the service availability retrieval fails.
        """
        if not self.remote:
            msg = "This method can only be executed when the client is in remote mode."
            raise RuntimeError(msg)

        request = submission_pb2.GetServiceStatusRequest(token=self.token, backend=self.backend)
        response: submission_pb2.GetServiceStatusResponse = self._call_rpc(self.stub.GetServiceStatus, request)
        if response.error.code:
            msg = f"Failed to get service status: {response.error.description} ({response.error.code})."
            raise RuntimeError(msg)

        return self._format_service_status_to_str(response.status), response.description

    def _run_local(self, circuit: CircuitRepr) -> SimulatorClientResult:
        from mqc3.client._local_simulator import local_run  # noqa:PLC0415

        execution_started_at = datetime.now(tz=self.timezone)
        local_result = local_run(self.n_shots, self.state_save_policy, circuit)
        execution_finished_at = datetime.now(tz=self.timezone)

        return SimulatorClientResult(
            execution_details=ExecutionDetails(
                simulator_version="",
                scheduler_version="",
                timeline=JobTimeline(
                    submitted_at=execution_started_at,
                    execution_started_at=execution_started_at,
                    execution_finished_at=execution_finished_at,
                    finished_at=execution_finished_at,
                ),
            ),
            _circuit_result=local_result.circuit_result,
            _machinery_result=None,
            _graph_result=None,
            states=local_result.states,
        )

    def run(self, representation: ReprType) -> SimulatorClientResult:
        """Run a representation with the remote or local simulator.

        Note:
            This method is synchronous in the sense that it submits a job and waits for the result.

        Note:
            If remote is ``False``, supports only circuit representations.

        Args:
            representation (ReprType): Representation.

        Returns:
            SimulatorClientResult: Result of the running.

        Raises:
            RuntimeError: If the service status is not available or failed to executing the job.
            ValueError: If in local mode and `representation` is not a `CircuitRepr`.

        Example:
            >>> from numpy import pi
            >>> from mqc3.client import SimulatorClient
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> from mqc3.circuit.state import BosonicState
            >>> client = SimulatorClient(n_shots=1, backend="cpu")
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.PhaseRotation(pi / 2)
            [QuMode(id=0)]
            >>> circuit.Q(0) | intrinsic.Measurement(0.0)
            Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
            >>> circuit.set_initial_state(0, BosonicState.vacuum())
            >>> result = client.run(circuit)
        """
        if not self.remote:
            if not isinstance(representation, CircuitRepr):
                msg = (
                    f"Local simulator supports only CircuitRepr (got {type(representation).__name__}). "
                    "Use remote execution or convert the input to CircuitRepr."
                )
                raise ValueError(msg)
            return self._run_local(representation)

        service_status, service_description = self.get_service_availability()
        if service_status != "AVAILABLE":
            msg = f"Service is not available (Current Status: {service_status}). Please try again later."
            if service_description:
                msg += f" Details: {service_description}"
            raise RuntimeError(msg)

        if service_description:
            sys.stdout.write(f"Notice: {service_description}\n")

        job_id = self.submit(representation)

        try:
            final_job_status, status_detail, _ = self.wait_for_completion(job_id)
        except RuntimeError as err:
            msg = f"An error raised while waiting for the job [job_id={job_id}] completion."
            raise RuntimeError(msg) from err

        if final_job_status != "COMPLETED":
            msg = f"Job [job_id={job_id}] failed with status {final_job_status}."
            if status_detail:
                msg += f" Details: {status_detail}"
            raise RuntimeError(msg)

        try:
            return self.get_job_result(job_id)
        except RuntimeError as err:
            msg = f"An error raised while retrieving the job [job_id={job_id}] result."
            raise RuntimeError(msg) from err
