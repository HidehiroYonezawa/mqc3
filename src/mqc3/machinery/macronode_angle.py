"""Machinery operation of continuous variable quantum computing."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING, TypeAlias

from mqc3.feedforward import FeedForward, Variable, feedforward, ff_to_add_constant
from mqc3.graph.program import ModeMeasuredVariable, to_pos_measured_variable
from mqc3.math import equiv_mod_pi
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphOperation as PbOperation
from mqc3.pb.mqc3_cloud.program.v1.machinery_pb2 import MachineryRepresentation as PbMachineryRepr

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mqc3.graph import GraphRepr
    from mqc3.graph import Operation as GraphOperation
    from mqc3.graph.ops import GraphOpParam


class MeasuredVariable(Variable):
    """Measured variable."""

    macronode_index: int
    abcd: int

    def __init__(self, macronode_index: int, abcd: int) -> None:
        """Initialize measured variable.

        Args:
            macronode_index (int): Index of the macronode.
            abcd (int): Index of the micronode in the macronode (a=0, b=1, c=2, d=3).
        """
        self.macronode_index = macronode_index
        self.abcd = abcd

    def get_from_operation(self) -> tuple[int, int]:
        """Get the index of the macronode and the index of the micronode in the macronode.

        Returns:
            tuple[int, int]: Index of the macronode, index of the micronode in the macronode (a=0, b=1, c=2, d=3).
        """
        return self.macronode_index, self.abcd


MachineOpParam: TypeAlias = FeedForward[MeasuredVariable] | float  # noqa: UP040


class MacronodeAngle:
    """Macronode angle."""

    def __init__(
        self,
        theta_a: MachineOpParam,
        theta_b: MachineOpParam,
        theta_c: MachineOpParam,
        theta_d: MachineOpParam,
    ) -> None:
        """Constructor.

        Args:
            theta_a (MachineOpParam): Measurement angle of micronode a.
            theta_b (MachineOpParam): Measurement angle of micronode b.
            theta_c (MachineOpParam): Measurement angle of micronode c.
            theta_d (MachineOpParam): Measurement angle of micronode d.

        Raises:
            TypeError: If one of the four input values is neither int nor float.
            ValueError: If the input four angles have invalid values.
        """
        try:
            for theta in (theta_a, theta_b, theta_c, theta_d):
                if isinstance(theta, FeedForward):
                    continue
                float(theta)  # pyright: ignore[reportUnusedExpression]
        except (ValueError, TypeError) as exc:
            message = "All elements must be able to be cast to float."
            raise TypeError(message) from exc

        if (
            not isinstance(theta_a, FeedForward)
            and not isinstance(theta_b, FeedForward)
            and not isinstance(theta_c, FeedForward)
            and not isinstance(theta_d, FeedForward)
        ):
            ab = equiv_mod_pi(theta_a, theta_b)
            bc = equiv_mod_pi(theta_b, theta_c)
            cd = equiv_mod_pi(theta_c, theta_d)
            if not (ab and bc and cd) and (ab or cd):
                msg = "The four angles must satisfy the following relation modulo pi.\n"
                msg += "`theta_a` == `theta_b` == `theta_c` == `theta_d` or\n"
                msg += "(`theta_a` != `theta_b` and `theta_c` != `theta_d`)."
                raise ValueError(msg)

        self._values: tuple[MachineOpParam, MachineOpParam, MachineOpParam, MachineOpParam] = (
            theta_a,
            theta_b,
            theta_c,
            theta_d,
        )

    def __iter__(self) -> Iterator[MachineOpParam]:
        """Iterator of the tuple (theta_a, theta_b, theta_c, theta_d).

        Returns:
            Iterator[MachineOpParam]: Iterator of the tuple (theta_a, theta_b, theta_c, theta_d).
        """
        return iter(self._values)

    def __getitem__(self, index: int) -> MachineOpParam:
        """Retrieve an angle from the tuple (theta_a, theta_b, theta_c, theta_d) based on the specified index.

        Args:
            index (int): Index of tuple (theta_a, theta_b, theta_c, theta_d).

        Returns:
            MachineOpParam: Measurement angle.
        """
        return self._values[index]

    def __len__(self) -> int:
        """Length of tuple (theta_a, theta_b, theta_c, theta_d).

        Returns:
            int: 4.
        """
        return len(self._values)

    def __eq__(self, other: MacronodeAngle) -> bool:
        """Check equality based on (theta_a, theta_b, theta_c, theta_d).

        Args:
            other (MacronodeAngle): The object to compare with this instance.

        Returns:
            bool: Whether this instance is equal to another instance based on the tuple
                (theta_a, theta_b, theta_c, theta_d).
        """
        return self._values == other._values

    def __str__(self) -> str:
        """String representation of the instance.

        Returns:
            str: String representation of the instance.
        """
        return str(self._values)

    def __repr__(self) -> str:
        """String representation of the instance.

        Returns:
            str: String representation of the instance.
        """
        return (
            f"MacronodeAngle(theta_a={self.theta_a}, "
            f"theta_b={self.theta_b}, "
            f"theta_c={self.theta_c}, "
            f"theta_d={self.theta_d})"
        )

    def __hash__(self) -> int:
        """Hash.

        Returns:
            int: Hash.
        """
        return hash(self._values)

    @property
    def theta_a(self) -> MachineOpParam:
        """Get theta_a."""
        return self._values[0]

    @property
    def theta_b(self) -> MachineOpParam:
        """Get theta_b."""
        return self._values[1]

    @property
    def theta_c(self) -> MachineOpParam:
        """Get theta_c."""
        return self._values[2]

    @property
    def theta_d(self) -> MachineOpParam:
        """Get theta_d."""
        return self._values[3]

    def is_measurable(self) -> bool:
        """Check if the macronode has angles that are equivalent to each other modulo pi.

        Note:
            If any of the angles is a feedforward, it is always considered measurable.

        Returns:
            bool: True if the macronode has angles that are equivalent to each other modulo pi.
        """
        if (
            isinstance(self.theta_a, FeedForward)
            or isinstance(self.theta_b, FeedForward)
            or isinstance(self.theta_c, FeedForward)
            or isinstance(self.theta_d, FeedForward)
        ):
            return True

        return (
            equiv_mod_pi(self.theta_a, self.theta_b)
            and equiv_mod_pi(self.theta_a, self.theta_c)
            and equiv_mod_pi(self.theta_a, self.theta_d)
            and equiv_mod_pi(self.theta_b, self.theta_c)
            and equiv_mod_pi(self.theta_c, self.theta_d)
        )

    def flatten(self) -> list[MachineOpParam]:
        """Flatten the macronode angle.

        Returns:
            list[MachineOpParam]: Flattened macronode angle.

        Example:
            >>> from mqc3.machinery.macronode_angle import MacronodeAngle
            >>> angle = MacronodeAngle(1, 2, 3, 4)
            >>> angle.flatten()
            [1, 2, 3, 4]
        """
        return [self.theta_a, self.theta_b, self.theta_c, self.theta_d]

    def has_feedforward(self) -> bool:
        """Check if the macronode angle has feedforward.

        Returns:
            bool: True if the macronode angle has feedforward.
        """
        return any(isinstance(theta, FeedForward) for theta in self.flatten())

    def proto(self) -> PbMachineryRepr.MacronodeAngle:  # noqa: D102
        def _default_if_ff(value: MachineOpParam, default: float) -> float:
            return value if not isinstance(value, FeedForward) else default

        return PbMachineryRepr.MacronodeAngle(
            theta_a=_default_if_ff(self.theta_a, 0.0),
            theta_b=_default_if_ff(self.theta_b, 0.0),
            theta_c=_default_if_ff(self.theta_c, 0.0),
            theta_d=_default_if_ff(self.theta_d, 0.0),
        )

    @staticmethod
    def construct_from_proto(proto: PbMachineryRepr.MacronodeAngle) -> MacronodeAngle:  # noqa: D102
        return MacronodeAngle(
            theta_a=proto.theta_a,
            theta_b=proto.theta_b,
            theta_c=proto.theta_c,
            theta_d=proto.theta_d,
        )


@feedforward
def _half_of(x: float) -> float:
    """Return half of x.

    Args:
        x (float): Input value.

    Returns:
        float: Half of x.
    """
    return x / 2.0


@feedforward
def _add_half_pi(x: float) -> float:
    """Return x + pi / 2.0.

    Args:
        x (float): Input value.

    Returns:
        float: x + pi / 2.0.
    """
    from math import pi  # noqa: PLC0415

    return x + pi / 2.0


@feedforward
def _atan_of(x: float) -> float:
    """Return atan(x).

    Args:
        x (float): Input value.

    Returns:
        float: atan(x).
    """
    from math import atan  # noqa: PLC0415

    return atan(x)


@feedforward
def _sub_from_half_pi(x: float) -> float:
    """Return pi / 2.0 - x.

    Args:
        x (float): Input value.

    Returns:
        float: pi / 2.0 - x.
    """
    from math import pi  # noqa: PLC0415

    return pi / 2.0 - x


@feedforward
def _minus_of(x: float) -> float:
    """Return -x.

    Args:
        x (float): Input value.

    Returns:
        float: -x.
    """
    return -x


@feedforward
def _add_quad_pi(x: float) -> float:
    """Return x + pi / 4.0.

    Args:
        x (float): Input value.

    Returns:
        float: x + pi / 4.0.
    """
    from math import pi  # noqa: PLC0415

    return x + pi / 4.0


@feedforward
def _exp_of_minus(x: float) -> float:
    """Return exp(-x).

    Args:
        x (float): Input value.

    Returns:
        float: exp(-x).
    """
    from math import exp  # noqa: PLC0415

    return exp(-x)


@feedforward
def _sub_quad_pi(x: float) -> float:
    """Return x - pi / 4.0.

    Args:
        x (float): Input value.

    Returns:
        float: x - pi / 4.0.
    """
    from math import pi  # noqa: PLC0415

    return x - pi / 4.0


@feedforward
def _acos_of(x: float) -> float:
    """Return acos(x).

    Args:
        x (float): Input value.

    Returns:
        float: acos(x).
    """
    from math import acos  # noqa: PLC0415

    return acos(x)


def from_measurement(theta: MachineOpParam) -> MacronodeAngle:
    """Get the macronode angle for measurement.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    return MacronodeAngle(theta, theta, theta, theta)


def from_initialization(theta: MachineOpParam) -> MacronodeAngle:
    """Get the macronode angle for initialization.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    return MacronodeAngle(theta, theta, theta, theta)


def from_phase_rotation(phi: MachineOpParam, *, swap: bool) -> MacronodeAngle:
    """Get the macronode angle for phase rotation.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    a = _half_of(phi)
    b = _add_half_pi(_half_of(phi))
    return MacronodeAngle(b, a, a, b) if swap else MacronodeAngle(a, b, a, b)


def from_shear_x_invariant(k: MachineOpParam, *, swap: bool) -> MacronodeAngle:
    """Get the macronode angle for x-invariant shear.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    a = _atan_of(k)
    b = pi / 2.0
    return MacronodeAngle(b, a, a, b) if swap else MacronodeAngle(a, b, a, b)


def from_shear_p_invariant(e: MachineOpParam, *, swap: bool) -> MacronodeAngle:
    """Get the macronode angle for p-invariant shear.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    a = 0
    b = _sub_from_half_pi(_atan_of(e))

    return MacronodeAngle(b, a, a, b) if swap else MacronodeAngle(a, b, a, b)


def from_squeeze(theta: MachineOpParam, *, swap: bool) -> MacronodeAngle:
    """Get the macronode angle for squeezing.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    a = _minus_of(theta)
    b = theta
    return MacronodeAngle(b, a, a, b) if swap else MacronodeAngle(a, b, a, b)


def from_45_squeeze(theta: MachineOpParam, *, swap: bool) -> MacronodeAngle:
    """Get the macronode angle for 45-degree squeezing.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    a = _add_quad_pi(_minus_of(theta))
    b = _add_quad_pi(theta)

    return MacronodeAngle(b, a, a, b) if swap else MacronodeAngle(a, b, a, b)


def from_arbitrary_first(
    _: MachineOpParam,
    beta: MachineOpParam,
    lam: MachineOpParam,
    *,
    swap: bool,
) -> MacronodeAngle:
    """Get the first macronode angle for arbitrary gate.

    Args:
        beta (MachineOpParam): Parameter `beta`.
        lam (MachineOpParam): Parameter `lam`.
        swap (bool): If swapped at the first macronode.

    Returns:
        MacronodeAngle: Macronode angle.

    Raises:
        TypeError: If both beta and lam are feedforward parameters.
    """
    if isinstance(beta, FeedForward) and isinstance(lam, FeedForward):
        msg = "Both beta and lam cannot be feedforward parameters."
        raise TypeError(msg)

    at_el = _atan_of(_exp_of_minus(lam))

    if isinstance(beta, FeedForward):
        assert not isinstance(at_el, FeedForward)  # noqa: S101
        a1 = ff_to_add_constant(-at_el)(beta)
        b1 = ff_to_add_constant(at_el)(beta)
    else:
        a1 = ff_to_add_constant(beta)(_minus_of(at_el))
        b1 = ff_to_add_constant(beta)(at_el)

    return MacronodeAngle(b1, a1, a1, b1) if swap else MacronodeAngle(a1, b1, a1, b1)


def from_arbitrary_second(
    alpha: MachineOpParam,
    beta: MachineOpParam,
    _: MachineOpParam,
    *,
    swap: bool,
) -> MacronodeAngle:
    """Get the second macronode angle for arbitrary gate.

    Args:
        alpha (MachineOpParam): Parameter `alpha`.
        beta (MachineOpParam): Parameter `beta`.
        swap (bool): Is swapped at the second macronode.

    Returns:
        MacronodeAngle: Macronode angles.

    Raises:
        TypeError: If both alpha and beta are feedforward parameters.
    """
    if isinstance(alpha, FeedForward) and isinstance(beta, FeedForward):
        msg = "Both alpha and beta cannot be feedforward parameters."
        raise TypeError(msg)

    if isinstance(alpha, FeedForward):
        assert not isinstance(beta, FeedForward)  # noqa: S101
        h_a_b = _half_of(ff_to_add_constant(-beta)(alpha))
    else:
        h_a_b = _half_of(ff_to_add_constant(alpha)(_minus_of(beta)))

    a2 = _add_quad_pi(h_a_b)
    b2 = _sub_quad_pi(h_a_b)
    return MacronodeAngle(b2, a2, a2, b2) if swap else MacronodeAngle(a2, b2, a2, b2)


def from_controlled_z(g: MachineOpParam, *, swap: bool) -> MacronodeAngle:
    """Get the macronode angle for controlled-Z gate.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    a = _atan_of(_half_of(g))
    b = pi / 2.0
    return MacronodeAngle(b, _minus_of(a), a, b) if swap else MacronodeAngle(_minus_of(a), b, a, b)


def from_beam_splitter(sqrt_r: MachineOpParam, theta_rel: MachineOpParam, *, swap: bool) -> MacronodeAngle:
    """Get the macronode angle for beam splitter.

    Returns:
        MacronodeAngle: Macronode angle.

    Raises:
        TypeError: If both sqrt_r and theta_rel are feedforward parameters.
    """
    if isinstance(sqrt_r, FeedForward) and isinstance(theta_rel, FeedForward):
        msg = "Both sqrt_r and theta_rel cannot be feedforward parameters."
        raise TypeError(msg)

    h_theta = _half_of(theta_rel)
    h_acos = _half_of(_acos_of(sqrt_r))

    if isinstance(h_theta, FeedForward):
        assert not isinstance(h_acos, FeedForward)  # noqa: S101
        alpha = ff_to_add_constant(h_acos)(h_theta)
        beta = ff_to_add_constant(-h_acos)(h_theta)
    else:
        alpha = ff_to_add_constant(h_theta)(h_acos)
        beta = ff_to_add_constant(h_theta)(_minus_of(h_acos))

    alpha_h_pi = _add_half_pi(alpha)
    beta_h_pi = _add_half_pi(beta)
    return (
        MacronodeAngle(alpha_h_pi, alpha, beta, beta_h_pi)
        if swap
        else MacronodeAngle(alpha, alpha_h_pi, beta, beta_h_pi)
    )


def from_two_mode_shear(a: MachineOpParam, b: MachineOpParam, *, swap: bool) -> MacronodeAngle:
    """Get the macronode angle for through two-mode shear.

    Args:
        a (MachineOpParam): Parameter `a`.
        b (MachineOpParam): Parameter `b`.
        swap (bool): If swapped.

    Returns:
        MacronodeAngle: Macronode angle.

    Raises:
        TypeError:  If both a and b are feedforward parameters.
    """
    if isinstance(a, FeedForward) and isinstance(b, FeedForward):
        msg = "Both a and b cannot be feedforward parameters."
        raise TypeError(msg)

    if isinstance(a, FeedForward):
        assert not isinstance(b, FeedForward)  # noqa: S101
        m = ff_to_add_constant(-b / 2.0)(a)
        p = ff_to_add_constant(b / 2.0)(a)
    else:
        m = ff_to_add_constant(a)(_minus_of(_half_of(b)))
        p = ff_to_add_constant(a)(_half_of(b))

    atan_m = _atan_of(m)
    atan_p = _atan_of(p)
    half_pi = pi / 2.0
    return (
        MacronodeAngle(half_pi, atan_m, atan_p, half_pi) if swap else MacronodeAngle(atan_m, half_pi, atan_p, half_pi)
    )


def from_manual(
    theta_a: MachineOpParam,
    theta_b: MachineOpParam,
    theta_c: MachineOpParam,
    theta_d: MachineOpParam,
    *,
    swap: bool,
) -> MacronodeAngle:
    """Get the macronode angle for manual settings.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    return (
        MacronodeAngle(theta_b, theta_a, theta_c, theta_d)
        if swap
        else MacronodeAngle(theta_a, theta_b, theta_c, theta_d)
    )


def from_wiring(*, swap: bool) -> MacronodeAngle:
    """Get the macronode angle for wiring.

    Args:
        swap (bool): If swapped.

    Returns:
        MacronodeAngle: Macronode angle.
    """
    h_pi = pi / 2.0
    return MacronodeAngle(h_pi, 0, 0, h_pi) if swap else MacronodeAngle(0, h_pi, 0, h_pi)


def convert_gm_param(param: GraphOpParam, graph: GraphRepr) -> MachineOpParam:
    """Convert GraphOpParam to MachineOpParam.

    Args:
        param (GraphOpParam): GraphOpParam.
        graph (GraphRepr): GraphRepr.

    Returns:
        MachineOpParam: MachineOpParam.

    Raises:
        ValueError: Invalid bd value.
    """
    if not isinstance(param, FeedForward):
        return param

    g_var = param.variable
    if isinstance(g_var, ModeMeasuredVariable):
        g_var = to_pos_measured_variable(g_var, graph)

    h, w, bd = g_var.get_from_operation()

    ind = graph.get_index(h, w)

    if bd == 0:
        abcd = 1
    elif bd == 1:
        abcd = 3
    else:
        msg = f"Invalid bd value: {bd} (0 or 1 is expected)."
        raise ValueError(msg)

    m_var = MeasuredVariable(ind, abcd)
    return param.func(m_var)


def from_graph_operation(op: GraphOperation, graph: GraphRepr) -> MacronodeAngle:  # noqa: C901, PLR0912
    """Get the macronode angle from a graph operation.

    Args:
        op (GraphOperation): Graph operation.
        graph (GraphRepr): Graph representation.

    Returns:
        MacronodeAngle | tuple[MacronodeAngle, MacronodeAngle]: Macronode angle.

    Raises:
        ValueError: Not supported arbitrary gate configuration.

    Example:
        >>> from mqc3.graph import GraphRepr, ops
        >>> from mqc3.machinery.macronode_angle import from_graph_operation
        >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
        >>> op = ops.Measurement((1, 2), theta=0)
        >>> from_graph_operation(op, graph)
        MacronodeAngle(theta_a=0, theta_b=0, theta_c=0, theta_d=0)
    """
    op_type = op.type()
    m_params = [convert_gm_param(param, graph) for param in op.parameters]

    if op_type == PbOperation.OPERATION_TYPE_MEASUREMENT:
        return from_measurement(*m_params)
    if op_type == PbOperation.OPERATION_TYPE_INITIALIZATION:
        return from_initialization(*m_params)
    if op_type == PbOperation.OPERATION_TYPE_PHASE_ROTATION:
        func = from_phase_rotation
    elif op_type == PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT:
        func = from_shear_x_invariant
    elif op_type == PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT:
        func = from_shear_p_invariant
    elif op_type == PbOperation.OPERATION_TYPE_SQUEEZING:
        func = from_squeeze
    elif op_type == PbOperation.OPERATION_TYPE_SQUEEZING_45:
        func = from_45_squeeze
    elif op_type == PbOperation.OPERATION_TYPE_CONTROLLED_Z:
        func = from_controlled_z
    elif op_type == PbOperation.OPERATION_TYPE_BEAM_SPLITTER:
        func = from_beam_splitter
    elif op_type == PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR:
        func = from_two_mode_shear
    elif op_type == PbOperation.OPERATION_TYPE_MANUAL:
        func = from_manual
    elif op_type == PbOperation.OPERATION_TYPE_WIRING:
        func = from_wiring
    elif op_type == PbOperation.OPERATION_TYPE_ARBITRARY_FIRST:
        func = from_arbitrary_first
    elif op_type == PbOperation.OPERATION_TYPE_ARBITRARY_SECOND:
        func = from_arbitrary_second
    else:
        msg = f"Unsupported operation type: {op_type}."
        raise ValueError(msg)

    return func(*m_params, swap=op.swap)
