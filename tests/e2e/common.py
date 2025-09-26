"""Common module for e2e tests."""

# pyright: reportUnusedExpression=false

from math import pi

import numpy as np

from mqc3.circuit import BosonicState, CircuitRepr, GaussianState
from mqc3.circuit.ops import intrinsic, std
from mqc3.client import MQC3Client
from mqc3.client.mqc3_client import BackendType
from mqc3.feedforward import feedforward


def construct_client(
    n_shots: int = 500,
    token: str | None = None,
    url: str | None = None,
    backend: BackendType = "emulator",
    *,
    ca_cert_path: str | None = None,
) -> MQC3Client:
    client = MQC3Client(
        n_shots=n_shots,
        backend=backend,
        ca_cert_file=ca_cert_path,
        run_timeout=3600,
    )
    if token is not None:
        client.token = token
    if url is not None:
        client.url = url

    return client


def circuit_only_intrinsics() -> CircuitRepr:
    c = CircuitRepr("circuit_only_intrinsics")
    c.Q(0) | intrinsic.PhaseRotation(pi / 4)
    c.Q(1) | intrinsic.PhaseRotation(3 * pi / 4) | intrinsic.Squeezing(pi / 4)
    c.Q(2) | intrinsic.Squeezing(pi / 4)
    c.Q(2, 1) | intrinsic.BeamSplitter(0.1, pi / 4)
    c.Q(1, 0) | intrinsic.ControlledZ(0.2)
    c.Q(0) | intrinsic.Measurement(0)
    c.Q(1) | intrinsic.Measurement(0)
    c.Q(2) | intrinsic.Measurement(0)
    return c


def circuit_with_std() -> CircuitRepr:
    c = CircuitRepr("circuit_with_std")
    c.Q(2, 1) | std.BeamSplitter(pi / 4, pi)
    c.Q(1, 0) | std.BeamSplitter(pi / 4, pi)
    c.Q(0) | intrinsic.Measurement(0)
    c.Q(1) | intrinsic.Measurement(0)
    c.Q(2) | intrinsic.Measurement(0)
    return c


def circuit_with_feedforward():
    @feedforward
    def displace_x(m: float) -> float:
        from math import sqrt  # noqa:PLC0415

        return sqrt(2) * m

    @feedforward
    def displace_p(m: float) -> float:
        from math import sqrt  # noqa:PLC0415

        return -sqrt(2) * m

    c = CircuitRepr("circuit_with_feedforward")
    c.Q(1, 2) | std.BeamSplitter(pi / 4, pi)
    c.Q(0, 1) | std.BeamSplitter(pi / 4, pi)
    m0 = c.Q(0) | intrinsic.Measurement(pi / 2.0)
    m1 = c.Q(1) | intrinsic.Measurement(0)
    c.Q(2) | intrinsic.Displacement(displace_x(m0), displace_p(m1))
    return c


def teleportation_circuit(disp: tuple[float, float], sq: float = 0.0):
    """Create a teleportation circuit."""
    circuit = circuit_with_feedforward()

    # Set initial states.
    for i in range(3):
        coeff = np.array([1.0 + 0.0j])
        mean = np.array([0.0 + 0.0j, 0.0 + 0.0j])
        if i == 0:
            mean = np.array([disp[0] + 0.0j, disp[1] + 0.0j])

        if i == 0:
            cov = np.array([[10**sq, 0.0], [0.0, 10 ** (-sq)]]) @ GaussianState.vacuum().cov
        elif i == 1:
            # p-squeezed
            cov = np.array([[1e8, 0.0], [0.0, 1e-8]]) @ GaussianState.vacuum().cov
        elif i == 2:
            # x-squeezed
            cov = np.array([[1e-8, 0.0], [0.0, 1e8]]) @ GaussianState.vacuum().cov

        gauss = GaussianState(mean, cov)
        circuit.set_initial_state(i, BosonicState(coeff, [gauss]))

    return circuit
