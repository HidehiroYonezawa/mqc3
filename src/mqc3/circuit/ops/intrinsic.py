"""Intrinsic operations of circuit representation."""

from abc import abstractmethod

from mqc3.circuit.ops._base import CircOpParam, MeasuredVariable, Operand, Operation
from mqc3.feedforward import FeedForward
from mqc3.math import equiv_mod_pi
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation


class Intrinsic(Operation):
    """Intrinsic operations."""

    def __init__(self) -> None:
        """Initialize the instance by calling the `Operation` class's initializer."""
        super().__init__()

    @abstractmethod
    def type(self) -> PbOperation.OperationType:
        """Get the operation type defined in the proto format."""

    def to_intrinsic_ops(self) -> list[Operation]:
        """Return itself."""
        return [self]

    def proto(self) -> list[PbOperation]:
        """Get proto format of the operation.

        Returns:
            list[PbOperation]: Proto format of the operation.
        """
        params = [p if not isinstance(p, FeedForward) else 0.0 for p in self.parameters()]
        return [PbOperation(type=self.type(), modes=self.opnd().get_ids(), parameters=params)]


class Measurement(Intrinsic):
    r"""Measure :math:`\hat{x} \sin \theta+\hat{p} \cos \theta`."""

    def __init__(self, theta: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            theta (CircOpParam): Operation parameter.
        """
        super().__init__()
        self.theta = theta

    def name(self) -> str:  # noqa: D102
        return "intrinsic.measurement"

    def n_modes(self) -> int:  # noqa: D102
        return 1

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.theta]

    def __ror__(self, opnd: Operand) -> MeasuredVariable:
        """Measure the input operand.

        Args:
            opnd (Operand): Operand to measure.

        Returns:
            MeasuredVariable: Measured variable (enable to use for feedforward).
        """
        super().__ror__(opnd)
        return MeasuredVariable(self)

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_MEASUREMENT

    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class Displacement(Intrinsic):
    r"""Displacement gate :math:`D\left(x, p\right)`.

    .. math::

        \binom{\hat{x}}{\hat{p}} \overset{\hat{D}\left(d_x, d_p\right)}{\longrightarrow}
        \binom{\hat{x}+d_x}{\hat{p}+d_p}
    """

    def __init__(self, x: CircOpParam, p: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            x (CircOpParam): Operation parameter `x`
            p (CircOpParam): Operation parameter `p`
        """
        super().__init__()
        self.x = x
        self.p = p

    def name(self) -> str:  # noqa: D102
        return "intrinsic.displacement"

    def n_modes(self) -> int:  # noqa: D102
        return 1

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.x, self.p]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_DISPLACEMENT

    def n_macronodes(self) -> int:  # noqa: D102
        return 0


class PhaseRotation(Intrinsic):
    r"""Phase rotation gate :math:`R(\phi)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, phi: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            phi (CircOpParam): Operation parameter.
        """
        super().__init__()
        self.phi = phi

    def name(self) -> str:  # noqa: D102
        return "intrinsic.phase_rotation"

    def n_modes(self) -> int:  # noqa: D102
        return 1

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.phi]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_PHASE_ROTATION

    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class ShearXInvariant(Intrinsic):
    r"""Shear gate (X-invariant) :math:`P(\kappa)`.

    .. math::

        \hat{P}^{\dagger}(\kappa)\binom{\hat{x}}{\hat{p}} \hat{P}(\kappa)=\left(\begin{array}{cc}
        1 & 0 \\
        2 \kappa & 1
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =P(\kappa)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, kappa: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            kappa (CircOpParam): Operation parameter.
        """
        super().__init__()
        self.kappa = kappa

    def name(self) -> str:  # noqa: D102
        return "intrinsic.shear_x_invariant"

    def n_modes(self) -> int:  # noqa: D102
        return 1

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.kappa]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT

    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class ShearPInvariant(Intrinsic):
    r"""Shear gate (P-invariant) :math:`Q(\eta)`.

    .. math::

        \hat{Q}^{\dagger}(\eta)\binom{\hat{x}}{\hat{p}} \hat{Q}(\eta)=\left(\begin{array}{cc}
        1 & 2 \eta \\
        0 & 1
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =Q(\eta)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, eta: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            eta (CircOpParam): Operation parameter.
        """
        super().__init__()
        self.eta = eta

    def name(self) -> str:  # noqa: D102
        return "intrinsic.shear_p_invariant"

    def n_modes(self) -> int:  # noqa: D102
        return 1

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.eta]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT

    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class Squeezing(Intrinsic):
    r"""Squeezing gate with some unusual phase rotation.

    Defined as :math:`R\left(-\frac{\pi}{2}\right) S_{\text{V}}(\cot \theta)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}} \\

    .. math::

        \hat{S}_\text{V}^{\dagger}(c)\binom{\hat{x}}{\hat{p}} \hat{S}_\text{V}(c)=\left(\begin{array}{cc}
        \frac{1}{c} & 0 \\
        0 & c
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S_\text{V}(c)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, theta: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            theta (CircOpParam): Operation parameter.

        Raises:
            ValueError: If an invalid value for `theta` is provided.
        """
        if not isinstance(theta, FeedForward) and equiv_mod_pi(theta, 0):
            msg = "`theta` must not be an integer multiple of pi."
            raise ValueError(msg)
        super().__init__()
        self.theta = theta

    def name(self) -> str:  # noqa: D102
        return "intrinsic.squeezing"

    def n_modes(self) -> int:  # noqa: D102
        return 1

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.theta]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SQUEEZING

    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class Squeezing45(Intrinsic):
    r"""45-degree squeezing gate.

    Defined as :math:`R\left(-\frac{\pi}{4}\right) S_{\text{V}}(\cot \theta) R\left(\frac{\pi}{4}\right)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}} \\

    .. math::

        \hat{S}_\text{V}^{\dagger}(c)\binom{\hat{x}}{\hat{p}} \hat{S}_\text{V}(c)=\left(\begin{array}{cc}
        \frac{1}{c} & 0 \\
        0 & c
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S_\text{V}(c)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, theta: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            theta (CircOpParam): Operation parameter.

        Raises:
            ValueError: If an invalid value for `theta` is provided.
        """
        if not isinstance(theta, FeedForward) and equiv_mod_pi(theta, 0):
            msg = "`theta` must not be an integer multiple of pi."
            raise ValueError(msg)
        super().__init__()
        self.theta = theta

    def name(self) -> str:  # noqa: D102
        return "intrinsic.squeezing45"

    def n_modes(self) -> int:  # noqa: D102
        return 1

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.theta]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SQUEEZING_45

    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class Arbitrary(Intrinsic):
    r"""Arbitrary gaussian gate :math:`R(\alpha) S(\lambda) R(\beta)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}} \\

    .. math::

        \hat{S}^{\dagger}(r)\binom{\hat{x}}{\hat{p}} \hat{S}(r)=\left(\begin{array}{cc}
        e^r & 0 \\
        0 & e^{-r}
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S(r)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(self, alpha: CircOpParam, beta: CircOpParam, lam: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            alpha (CircOpParam): Operation parameter `alpha`.
            beta (CircOpParam): Operation parameter `beta`.
            lam (CircOpParam): Operation parameter `lam`.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

    def name(self) -> str:  # noqa: D102
        return "intrinsic.arbitrary"

    def n_modes(self) -> int:  # noqa: D102
        return 1

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.alpha, self.beta, self.lam]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_ARBITRARY

    def n_macronodes(self) -> int:  # noqa: D102
        return 2


class ControlledZ(Intrinsic):
    r"""Controlled-Z gate :math:`C_Z(g)`.

    .. math::

        \hat{C}_Z^{\dagger}(g)\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right) \hat{C}_Z(g)=
        \left(\begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & g & 1 & 0 \\
        g & 0 & 0 & 1
        \end{array}\right)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        =C_Z(g)\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
    """

    def __init__(self, g: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            g (CircOpParam): Operation parameter.
        """
        super().__init__()
        self.g = g

    def name(self) -> str:  # noqa: D102
        return "intrinsic.controlled_z"

    def n_modes(self) -> int:  # noqa: D102
        return 2

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.g]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_CONTROLLED_Z

    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class BeamSplitter(Intrinsic):
    r"""Beam splitter interaction with some unusual phase rotation.

    Defined as :math:`B\left(\sqrt{R}, \theta_{\text{rel}}\right)`.

    .. math::

        \hat{B}^{\dagger}(\sqrt{R}, \theta_\text{rel})\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right) \hat{B}(\sqrt{R}, \theta_\text{rel})
        =
        \left(\begin{array}{rrrr}
        \cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) &
        -\sin (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha-\beta) \cos (\alpha+\beta) \\
        \sin (\alpha+\beta) \sin (\alpha-\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta) &
        \sin (\alpha-\beta) \cos (\alpha+\beta) & -\sin (\alpha+\beta) \cos (\alpha-\beta) \\
        \sin (\alpha+\beta) \cos (\alpha-\beta) & -\sin (\alpha-\beta) \cos (\alpha+\beta) &
        \cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) \\
        -\sin (\alpha-\beta) \cos (\alpha+\beta) & \sin (\alpha+\beta) \cos (\alpha-\beta) &
        \sin (\alpha+\beta) \sin (\alpha-\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta)
        \end{array}\right)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        =B(\sqrt{R}, \theta_\text{rel})\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)

    .. math::

        \begin{align}
        \alpha &= \frac{\theta_\text{rel} + \operatorname{arccos} \sqrt{R}}{2} \\
        \beta &= \frac{\theta_\text{rel} - \operatorname{arccos} \sqrt{R}}{2}
        \end{align}
    """

    def __init__(self, sqrt_r: CircOpParam, theta_rel: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            sqrt_r (CircOpParam): Operation parameter `sqrt_r`.
            theta_rel (CircOpParam): Operation parameter `theta_rel`.

        Raises:
            ValueError: If an invalid value for `sqrt_r` is input.
        """
        if not isinstance(sqrt_r, FeedForward) and (sqrt_r < 0 or sqrt_r > 1):
            msg = "`sqrt_r` must be in the range [0, 1]."
            raise ValueError(msg)

        super().__init__()
        self.sqrt_r = sqrt_r
        self.theta_rel = theta_rel

    def name(self) -> str:  # noqa: D102
        return "intrinsic.beam_splitter"

    def n_modes(self) -> int:  # noqa: D102
        return 2

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.sqrt_r, self.theta_rel]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_BEAM_SPLITTER

    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class TwoModeShear(Intrinsic):
    r"""Two-mode Shear :math:`P_2(a, b)`.

    .. math::

        \hat{P_2}^{\dagger}(a, b)\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right) \hat{P_2}(a, b)=
        \left(\begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        2 a & b & 1 & 0 \\
        b & 2 a & 0 & 1
        \end{array}\right)
        \left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
        =P_2(a, b)\left(\begin{array}{c}
        \hat{x}_1 \\
        \hat{x}_2 \\
        \hat{p}_1 \\
        \hat{p}_2
        \end{array}\right)
    """

    def __init__(self, a: CircOpParam, b: CircOpParam) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            a (CircOpParam): Operation parameter `a`.
            b (CircOpParam): Operation parameter `b`.
        """
        super().__init__()
        self.a = a
        self.b = b

    def name(self) -> str:  # noqa: D102
        return "intrinsic.two_mode_shear"

    def n_modes(self) -> int:  # noqa: D102
        return 2

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.a, self.b]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR

    def n_macronodes(self) -> int:  # noqa: D102
        return 1


class Manual(Intrinsic):
    """Interaction with manually specified four homodyne angles."""

    def __init__(
        self,
        theta_a: CircOpParam,
        theta_b: CircOpParam,
        theta_c: CircOpParam,
        theta_d: CircOpParam,
    ) -> None:
        """Initialize the instance by calling the `Intrinsic` class's initializer.

        Args:
            theta_a (CircOpParam): Operation parameter
            theta_b (CircOpParam): Operation parameter
            theta_c (CircOpParam): Operation parameter
            theta_d (CircOpParam): Operation parameter

        Raises:
            ValueError: If `theta_a` is equal to `theta_b` modulo pi, or `theta_c` is equal to `theta_d` modulo pi.
        """
        if (
            not isinstance(theta_a, FeedForward)
            and not isinstance(theta_b, FeedForward)
            and equiv_mod_pi(theta_a, theta_b)
        ):
            msg = "`theta_a` must not be equal to `theta_b` modulo pi."
            raise ValueError(msg)
        if (
            not isinstance(theta_c, FeedForward)
            and not isinstance(theta_d, FeedForward)
            and equiv_mod_pi(theta_c, theta_d)
        ):
            msg = "`theta_c` must not be equal to `theta_d` modulo pi."
            raise ValueError(msg)
        super().__init__()
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.theta_c = theta_c
        self.theta_d = theta_d

    def name(self) -> str:  # noqa: D102
        return "intrinsic.manual"

    def n_modes(self) -> int:  # noqa: D102
        return 2

    def parameters(self) -> list[CircOpParam]:  # noqa: D102
        return [self.theta_a, self.theta_b, self.theta_c, self.theta_d]

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_MANUAL

    def n_macronodes(self) -> int:  # noqa: D102
        return 1
