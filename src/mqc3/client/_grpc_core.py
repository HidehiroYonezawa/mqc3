import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import grpc

RETRYABLE_CODES = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}


class GrpcConnectionManager(ABC):
    """Abstract base class for gRPC clients.

    Handles channel creation (secure or insecure), retries, reconnection, and basic gRPC error classification.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        address: str,
        max_send_message_length: int,
        max_receive_message_length: int,
        max_retry_count: int,
        timeout_sec: float,
        secure_remote: bool = True,
        ca_cert_file: str | None,
    ) -> None:
        """Initialize the gRPC client.

        Args:
            address (str): The gRPC server address.
            max_send_message_length (int): Maximum size (in bytes) for outbound messages.
            max_receive_message_length (int): Maximum size (in bytes) for inbound messages.
            max_retry_count (int): Number of times to retry before reconnecting.
            timeout_sec (float): Default RPC timeout in seconds.
            secure_remote (bool): Whether to use TLS for secure communication.
            ca_cert_file (str | None): Path to CA certificate file for TLS, or None for insecure.

        Raises:
            FileNotFoundError: If the CA certificate file is provided but not found.
        """
        self.address = address
        self.options = [
            ("grpc.max_send_message_length", max_send_message_length),
            ("grpc.max_receive_message_length", max_receive_message_length),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.keepalive_permit_without_calls", 1),
        ]
        self.max_retry_count = max_retry_count
        self.timeout_sec = timeout_sec

        self.secure_remote = secure_remote
        self.ca_cert_file = ca_cert_file
        self.ca_cert = None
        if ca_cert_file is not None:
            try:
                with Path(ca_cert_file).open("rb") as f:
                    ca_cert = f.read()
            except FileNotFoundError as err:
                msg = f"CA certificate file not found: {ca_cert_file}"
                raise FileNotFoundError(msg) from err

            self.ca_cert = ca_cert

        self.channel = self._create_channel()
        self._recreate_stub()

    @property
    def max_send_message_length(self) -> int:
        """Maximum size (in bytes) for outbound messages."""
        return self.options[0][1]

    @property
    def max_receive_message_length(self) -> int:
        """Maximum size (in bytes) for inbound messages."""
        return self.options[1][1]

    def _create_channel(self) -> grpc.Channel:
        """Create a gRPC channel (secure or insecure depending on configuration).

        Returns:
            grpc.Channel: The configured gRPC channel.
        """
        if self.secure_remote:
            credentials = grpc.ssl_channel_credentials(root_certificates=self.ca_cert)
            return grpc.secure_channel(self.address, credentials, options=self.options)

        # Not secure remote
        return grpc.insecure_channel(self.address, options=self.options)

    def close(self) -> None:
        """Close the gRPC channel."""
        self.channel.close()

    def reconnect(self) -> None:
        """Reconnect by closing and recreating the channel and stub.

        This method can be called when repeated gRPC calls fail.
        """
        sys.stdout.write("Reconnecting gRPC channel...\n")
        self.close()
        self.channel = self._create_channel()
        self._recreate_stub()

    @abstractmethod
    def _recreate_stub(self) -> None:
        """Recreate the gRPC stub after reconnecting the channel.

        This method must be implemented by subclasses.
        """
        msg = "Subclasses must implement stub recreation."
        raise NotImplementedError(msg)

    def _call_rpc(self, func: Callable[[Any], Any], request: Any, *, timeout_sec: float | None = None) -> Any:  # noqa: ANN401
        """Call an RPC with retry and reconnect logic.

        Args:
            func (Callable): The gRPC stub method to call.
            request (Any): The request message to send.
            timeout_sec (float | None): Optional timeout in seconds; uses default if None.

        Returns:
            Any: The gRPC response object.
        """
        timeout: float = timeout_sec if timeout_sec is not None else self.timeout_sec

        if self.max_retry_count > 1:
            for _attempt in range(self.max_retry_count - 1):
                try:
                    return func(request, timeout=timeout)  # type: ignore  # noqa: PGH003
                except grpc.RpcError as e:
                    if e.code() in RETRYABLE_CODES:
                        self._handle_rpc_error(e, just_logging=True)
                    else:
                        self._handle_rpc_error(e, just_logging=False)

            self.reconnect()

        try:
            return func(request, timeout=timeout)  # type: ignore  # noqa: PGH003
        except grpc.RpcError as e:
            self._handle_rpc_error(e, just_logging=False)

    def _handle_rpc_error(self, err: grpc.RpcError, *, just_logging: bool) -> None:
        """Handle a gRPC RpcError by logging and optionally raising a categorized error.

        Args:
            err (grpc.RpcError): The gRPC exception to handle.
            just_logging (bool): If True, only log the error and return silently.

        Raises:
            RuntimeError: If UNAVAILABLE or unknown gRPC errors.
            TimeoutError: If DEADLINE_EXCEEDED.
            ValueError: If NOT_FOUND.
        """
        code = err.code()
        msg = f"code={code.name}, detail={err.details()}"
        if just_logging:
            sys.stdout.write(f"{msg}\n")
        else:
            sys.stderr.write(f"{msg}\n")

        if just_logging:
            return

        if code == grpc.StatusCode.UNAVAILABLE:
            msg = "Unable to connect to gRPC server: UNAVAILABLE"
            raise RuntimeError(msg) from err
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            msg = "gRPC request timed out"
            raise TimeoutError(msg) from err
        if code == grpc.StatusCode.NOT_FOUND:
            msg = "Requested resource not found"
            raise ValueError(msg) from err
        msg = f"Unexpected gRPC error: {code.name} - {err.details()}"
        raise RuntimeError(msg) from err
