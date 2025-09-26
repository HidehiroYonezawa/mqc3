"""Locally run circuit representation."""


# pyright: reportUnusedExpression=false

from __future__ import annotations

try:
    import strawberryfields as sf
    from strawberryfields import ops as sf_ops
except ImportError as e:
    msg = """StrawberryFields is not installed.
Please install mqc3 with the sf option using:

pip install '<path/to/mqc3>[sf]'

or install StrawberryFields separately with:

pip install strawberryfields

"""
    raise ImportError(msg) from e


import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import acos, atan, log, pi, tan
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np

import mqc3.circuit.ops.intrinsic as intrinsic_ops
import mqc3.circuit.ops.std as std_ops
from mqc3.circuit.result import CircuitOperationMeasuredValue, CircuitResult, CircuitShotMeasuredValue
from mqc3.circuit.state import BosonicState, GaussianState
from mqc3.constant import hbar
from mqc3.feedforward import FeedForward

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqc3.circuit.program import CircOpParam, CircuitRepr

logger = logging.getLogger(__name__)


def __get_value(q, param: CircOpParam) -> float:  # noqa:ANN001
    if isinstance(param, float | int):
        return param
    if isinstance(param, FeedForward):
        func = param.func
        mode = param.variable.get_from_operation().opnd().get_ids()[0]
        return func(q[mode].par)  # pyright: ignore[reportReturnType]
    msg = f"Invalid parameter: {param}"
    raise ValueError(msg)


def __add_std_squeezing(program: sf.Program, squeezing: std_ops.Squeezing) -> None:
    with program.context as q:
        inds = squeezing.opnd().get_ids()
        r = __get_value(q, squeezing.r)
        sf_ops.Sgate(r) | q[inds[0]]


def __add_std_bs(program: sf.Program, bs: std_ops.BeamSplitter) -> None:
    with program.context as q:
        inds = bs.opnd().get_ids()
        theta = __get_value(q, bs.theta)
        phi = __get_value(q, bs.phi)
        sf_ops.BSgate(theta, phi) | (q[inds[0]], q[inds[1]])


def __add_intrinsic_op(  # noqa: C901,PLR0912,PLR0914,PLR0915
    program: sf.Program, mqc3_op: intrinsic_ops.Intrinsic
) -> None:
    inds = mqc3_op.opnd().get_ids()

    if isinstance(mqc3_op, intrinsic_ops.Measurement):
        with program.context as q:
            theta = __get_value(q, mqc3_op.theta)
            sf_ops.MeasureHomodyne(pi / 2.0 - theta) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.Displacement):
        with program.context as q:
            x = __get_value(q, mqc3_op.x)
            p = __get_value(q, mqc3_op.p)
            sf_ops.Xgate(x) | q[inds[0]]
            sf_ops.Zgate(p) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.PhaseRotation):
        with program.context as q:
            phi = __get_value(q, mqc3_op.phi)
            sf_ops.Rgate(phi) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.ShearXInvariant):
        with program.context as q:
            kappa = __get_value(q, mqc3_op.kappa)
            sf_ops.Pgate(kappa * 2.0) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.ShearPInvariant):
        with program.context as q:
            eta = __get_value(q, mqc3_op.eta)
            if eta == 0:
                acot_eta = pi / 2.0
            elif eta > 0:
                acot_eta = atan(1 / eta)
            elif eta < 0:
                acot_eta = atan(1 / eta) + pi
            half_pi = pi / 2.0
            half_theta = acot_eta / 2.0
            sign = sf.math.sign(tan(half_theta))
            sf_ops.Rgate(half_theta) | q[inds[0]]
            sf_ops.Sgate(r=-log(sign * tan(half_theta))) | q[inds[0]]
            sf_ops.Rgate(half_theta - sign * half_pi) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.Squeezing):
        with program.context as q:
            theta = __get_value(q, mqc3_op.theta)
            sign = sf.math.sign(tan(theta))
            sf_ops.Sgate(r=log(sign * tan(theta))) | q[inds[0]]
            sf_ops.Rgate(-sign * pi / 2.0) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.Squeezing45):
        with program.context as q:
            theta = __get_value(q, mqc3_op.theta)
            sign = sf.math.sign(tan(theta))
            sf_ops.Rgate(pi / 4.0) | q[inds[0]]
            sf_ops.Sgate(r=log(sign * tan(theta))) | q[inds[0]]
            if sign > 0:
                sf_ops.Rgate(-pi / 4.0) | q[inds[0]]
            else:
                sf_ops.Rgate(3 * pi / 4.0) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.Arbitrary):
        with program.context as q:
            alpha = __get_value(q, mqc3_op.alpha)
            beta = __get_value(q, mqc3_op.beta)
            lam = __get_value(q, mqc3_op.lam)
            sf_ops.Rgate(beta) | q[inds[0]]
            sf_ops.Sgate(lam) | q[inds[0]]
            sf_ops.Rgate(alpha) | q[inds[0]]
    elif isinstance(mqc3_op, intrinsic_ops.ControlledZ):
        with program.context as q:
            g = __get_value(q, mqc3_op.g)
            sf_ops.CZgate(g) | (q[inds[0]], q[inds[1]])  # pyright: ignore[reportArgumentType]
    elif isinstance(mqc3_op, intrinsic_ops.BeamSplitter):
        with program.context as q:
            sqrt_r = __get_value(q, mqc3_op.sqrt_r)
            theta_rel = __get_value(q, mqc3_op.theta_rel)
            h_theta = theta_rel / 2.0
            h_acos = acos(sqrt_r) / 2.0
            alpha = h_theta + h_acos
            beta = h_theta - h_acos
            sf_ops.BSgate() | (q[inds[0]], q[inds[1]])
            sf_ops.Rgate(alpha * 2) | q[inds[0]]
            sf_ops.Rgate(beta * 2) | q[inds[1]]
            sf_ops.BSgate() | (q[inds[1]], q[inds[0]])
    elif isinstance(mqc3_op, intrinsic_ops.TwoModeShear):
        with program.context as q:
            a = __get_value(q, mqc3_op.a)
            b = __get_value(q, mqc3_op.b)
            sf_ops.CZgate(b) | (q[inds[0]], q[inds[1]])  # pyright: ignore[reportArgumentType]
            sf_ops.Pgate(a * 2.0) | q[inds[0]]
            sf_ops.Pgate(a * 2.0) | q[inds[1]]
    else:
        msg = f"Unsupported operation type: {mqc3_op.name()}."
        raise TypeError(msg)


def _convert_to_program(circuit: CircuitRepr) -> sf.Program:
    program = sf.Program(circuit.n_modes)

    # Set initial state.
    for index in range(circuit.n_modes):
        state = circuit.get_initial_state(index)
        if not isinstance(state, BosonicState):
            msg = "Initial state must be 'BosonicState' instance."
            raise TypeError(msg)
        if state.n_peaks != 1:
            msg = "Only single-peak initial states are supported."
            raise ValueError(msg)
        gaussian = state.get_gaussian_state(0)
        with program.context as q:
            sf_ops.Gaussian(gaussian.cov, gaussian.mean) | q[index]

    # Set operations.
    for mqc3_op in circuit:
        if isinstance(mqc3_op, std_ops.Squeezing):
            __add_std_squeezing(program, mqc3_op)
        elif isinstance(mqc3_op, std_ops.BeamSplitter):
            __add_std_bs(program, mqc3_op)
        else:
            for intrinsic in mqc3_op.to_intrinsic_ops():
                if not isinstance(intrinsic, intrinsic_ops.Intrinsic):
                    msg = "Intrinsic must be an instance of Intrinsic."
                    raise TypeError(msg)
                __add_intrinsic_op(program, intrinsic)
    return program


def _convert_to_circuit_result(sf_results: list[NDArray[np.float64] | None]) -> CircuitResult:
    result = CircuitResult(shot_measured_values=[])
    for sf_shot_result in sf_results:
        shot_measured_value: list[CircuitOperationMeasuredValue] = []
        if sf_shot_result is not None:
            for index, value in enumerate(sf_shot_result):
                if value is not None:
                    shot_measured_value.append(CircuitOperationMeasuredValue(index, value))
        result.measured_vals.append(CircuitShotMeasuredValue(shot_measured_value))
    return result


def _convert_to_bosonic_state(
    sf_state: sf.backends.states.BaseBosonicState,  # pyright: ignore[reportAttributeAccessIssue]
) -> BosonicState:
    # SF base  : xpxpxpxp
    # MQC3 base: xxxxpppp
    n = sf_state.num_modes
    return BosonicState(
        np.array(sf_state.weights(), dtype=np.complex128),
        [
            GaussianState(
                # Reorder mean vector from (xpxpxpxp...) to (xxxxpppp...)
                np.array(sf_state.means()[i], dtype=np.complex128).reshape(n, 2).T.flatten(),
                # Reorder covariance matrix from (xpxpxpxp...) to (xxxxpppp...) using transpose
                np.array(sf_state.covs()[i], dtype=np.float64)
                .reshape(n, 2, n, 2)
                .transpose(1, 0, 3, 2)
                .reshape(n * 2, n * 2),
            )
            for i in range(sf_state.num_weights)
        ],
    )


@dataclass(frozen=True)
class LocalResult:
    """The result of executing a quantum circuit."""

    execution_time: timedelta
    """The time to execute a quantum circuit."""

    circuit_result: CircuitResult
    """Measurement results after circuit execution."""

    states: list[BosonicState]
    """States after circuit execution."""


def local_run(n_shots: int, state_save_policy: str, circuit: CircuitRepr) -> LocalResult:
    sf.hbar = hbar
    program = _convert_to_program(circuit)

    started_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    results: list[NDArray[np.float64] | None] = []
    states: list[BosonicState] = []
    for i in range(n_shots):
        engine = sf.Engine(backend="bosonic")
        result = engine.run(program)
        if result.samples is not None and len(result.samples) > 0:
            results.append(result.samples[0])
        else:
            results.append(None)
        if state_save_policy == "all" or (i == 0 and state_save_policy == "first_only"):
            states.append(_convert_to_bosonic_state(result.state))
    finished_at = datetime.now(started_at.tzinfo)

    return LocalResult(
        execution_time=finished_at - started_at,
        circuit_result=_convert_to_circuit_result(results),
        states=states,
    )
