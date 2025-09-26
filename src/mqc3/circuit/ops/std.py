"""Standard operations of circuit representation."""

from math import pi

from mqc3.circuit.ops import intrinsic
from mqc3.circuit.ops._base import CircOpParam, Operand, Operation
from mqc3.feedforward import FeedForward, feedforward, ff_to_add_constant
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation

__all__ = ["BeamSplitter", "Squeezing"]


class Squeezing(Operation):
    r"""Squeezing gate :math:`S(r)`.

    .. math::

        \hat{S}^{\dagger}(r)\binom{\hat{x}}{\hat{p}} \hat{S}(r)=\left(\begin{array}{cc}
        e^{-r} & 0 \\
        0 & e^r
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S(r)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, r: CircOpParam) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            r (CircOpParam): Squeezing parameter.
        """
        super().__init__()
        self.r = r

    def name(self) -> str:  # noqa: D102
        return "std.squeezing"

    def n_modes(self) -> int:  # noqa: D102
        return 1

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.r]

    def to_intrinsic_ops(self) -> list[Operation]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Returns:
            list[Operation]: Equivalent sequence of the intrinsic operations.
        """
        if isinstance(self.r, FeedForward):

            @feedforward
            def mul_plus(x: float) -> float:
                return x

            intrinsic_ops = [intrinsic.Arbitrary(alpha=0, beta=0, lam=mul_plus(self.r))]
        else:
            intrinsic_ops = [intrinsic.Arbitrary(alpha=0, beta=0, lam=self.r)]

        for op in intrinsic_ops:
            op._opnd = self._opnd  # noqa: SLF001
        return intrinsic_ops  # pyright: ignore[reportReturnType]

    def n_macronodes(self) -> int:  # noqa: D102
        return 2

    def proto(self) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op in self.to_intrinsic_ops():
            ret += op.proto()
        return ret


class BeamSplitter(Operation):
    r"""Beam splitter interaction :math:`B_\text{std}(\theta, \phi)`.

    .. math::

        \begin{align}
        \hat{B}_\text{std}^{\dagger}(\theta, \phi)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        \hat{B}_\text{std}(\theta, \phi)
        &=
        \left(\begin{array}{cccc}
        \cos \theta & -\sin \theta \cos \phi & 0 & -\sin \theta \sin \phi \\
        \sin \theta \cos \phi & \cos \theta & -\sin \theta \sin \phi & 0 \\
        0 & \sin \theta \sin \phi & \cos \theta & -\sin \theta \cos \phi \\
        \sin \theta \sin \phi & 0 & \sin \theta \cos \phi & \cos \theta
        \end{array}\right)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right) \\
        &=
        B_\text{std}(\theta, \phi)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        \end{align}
    """

    def __init__(self, theta: CircOpParam, phi: CircOpParam) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            theta (CircOpParam): Operation parameter `theta`.
            phi (CircOpParam): Operation parameter `phi`.
        """
        super().__init__()
        self.theta = theta
        self.phi = phi

    def name(self) -> str:  # noqa: D102
        return "std.beam_splitter"

    def n_modes(self) -> int:  # noqa: D102
        return 2

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.theta, self.phi]

    def to_intrinsic_ops(self) -> list[Operation]:
        """Convert the operation into an equivalent sequence of the intrinsic operations.

        Returns:
            list[Operation]: Equivalent sequence of the intrinsic operations.

        Raises:
            ValueError: If the `_opnd` attribute is `None`.
            TypeError: If both phi and theta are feedforward parameters.
        """
        if isinstance(self.phi, FeedForward) and isinstance(self.theta, FeedForward):
            msg = "Both phi and theta cannot be feedforward parameters."
            raise TypeError(msg)

        operands = self._opnd
        if operands is None:
            msg = "Operands are not set."
            raise ValueError(msg)
        first_opnd = Operand(modes=operands.modes[0:1], _program=operands._program)  # noqa: SLF001
        second_opnd = Operand(modes=operands.modes[1:2], _program=operands._program)  # noqa: SLF001

        @feedforward
        def sub_pi(x: float) -> float:
            from math import pi  # noqa: PLC0415

            return x - pi

        before1 = intrinsic.PhaseRotation(sub_pi(self.phi))
        before1._opnd = first_opnd  # noqa: SLF001

        before2 = intrinsic.PhaseRotation(-pi / 2.0)
        before2._opnd = second_opnd  # noqa: SLF001

        @feedforward
        def add_half_pi(x: float) -> float:
            from math import pi  # noqa: PLC0415

            return x + pi / 2.0

        bs = intrinsic.Manual(theta_a=0, theta_b=pi / 2, theta_c=self.theta, theta_d=add_half_pi(self.theta))
        bs._opnd = self._opnd  # noqa: SLF001

        if isinstance(self.phi, FeedForward):
            assert not isinstance(self.theta, FeedForward)  # noqa: S101
            ff = self.phi
            cons = self.theta
        else:
            ff = self.theta
            cons = self.phi

        @feedforward
        def make_after1_phi(x: float) -> float:
            from math import pi  # noqa: PLC0415

            return -x + pi

        after1 = intrinsic.PhaseRotation(ff_to_add_constant(-cons)(make_after1_phi(ff)))
        after1._opnd = first_opnd  # noqa: SLF001

        @feedforward
        def make_after2_phi(x: float) -> float:
            from math import pi  # noqa: PLC0415

            return -x + pi / 2.0

        after2 = intrinsic.PhaseRotation(make_after2_phi(ff))
        after2._opnd = second_opnd  # noqa: SLF001

        return [before1, before2, bs, after1, after2]

    def n_macronodes(self) -> int:  # noqa: D102
        return 5

    def proto(self) -> list[PbOperation]:  # noqa: D102
        ret = []
        for op in self.to_intrinsic_ops():
            ret += op.proto()
        return ret
