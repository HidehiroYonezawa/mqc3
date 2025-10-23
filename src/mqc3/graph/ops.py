"""Operations of graph representation."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from mqc3.feedforward import FeedForward, Variable
from mqc3.math import equiv_mod_pi
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphOperation as PbOperation


class PosMeasuredVariable(Variable):
    """Measured variable specified by the position in the graph representation."""

    h: int
    w: int
    bd: int

    def __init__(self, h: int, w: int, bd: int) -> None:
        """Initialize the measured variable.

        Args:
            h (int): The first argument of the coordinates in the graph representation.
            w (int): The second argument of the coordinates in the graph representation.
            bd (int): 0 or 1 (b or d).
        """
        self.h = h
        self.w = w
        self.bd = bd

    def get_from_operation(self) -> tuple[int, int, int]:
        """Get the measured value from the operation.

        Returns:
            tuple[int, int, int]: Pair of the macronode position and the measured micronode.
        """
        return (self.h, self.w, self.bd)


class ModeMeasuredVariable(Variable):
    """Measured variable specified by the mode number."""

    mode: int

    def __init__(self, mode: int) -> None:
        """Initialize the measured variable.

        Args:
            mode (int): The mode number.
        """
        self.mode = mode

    def get_from_operation(self) -> int:
        """Get the measured value from the operation.

        Returns:
            int: The mode number.
        """
        return self.mode


GraphOpParam: TypeAlias = FeedForward[PosMeasuredVariable | ModeMeasuredVariable] | float  # noqa: UP040


@dataclass
class Operation:
    """Operation class in graph representation."""

    macronode: tuple[int, int] = (-1, -1)
    "macronode used in the operation."

    parameters: list[GraphOpParam] = field(default_factory=list)
    "list of parameters of the operation."

    swap: bool = False
    "`True` if the output modes of macronode are swapped."

    initialized_modes: list[int] = field(default_factory=list)
    "modes initialized by the operation."

    displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0.0, 0.0)
    "Displacement (x, p) applied between the first macronode of the operation and the up one."

    displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0.0, 0.0)
    "Displacement (x, p) applied between the first macronode of the operation and the left one."

    readout: bool = False
    "readout the measured values in the macronode for the operation."

    @abstractmethod
    def type(self) -> PbOperation.OperationType:
        """Abstract method to return the operation type defined in the proto format.

        Returns:
            PbOperation.OperationType: Operation type defined in the proto format.
        """

    @abstractmethod
    def _get_init_args(self) -> dict[str, Any]:
        """Abstract method to return the initial arguments of the operation.

        Returns:
            dict[str, Any]: Initial arguments of the operation.
        """


class Wiring(Operation):
    """Wire."""

    def __init__(
        self,
        macronode: tuple[int, int],
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Construct a wire operation.

        Args:
            macronode (tuple[int, int]): Position of the macronode.
            swap (bool): Swap the output modes of the macronode.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Whether to readout the measured values.

        Examples:
            >>> from mqc3.graph import Wiring
            >>> wire = Wiring((1, 2), swap=False)
        """
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:
        """Get the operation type defined in the proto format.

        Returns:
            PbOperation.OperationType: Operation type defined in the proto format.
        """
        return PbOperation.OPERATION_TYPE_WIRING

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "swap": self.swap,
            "readout": self.readout,
        }


class Measurement(Operation):
    r"""Measure :math:`\hat{x} \sin \theta+\hat{p} \cos \theta`."""

    def __init__(
        self,
        macronode: tuple[int, int],
        theta: GraphOpParam,
        *,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = True,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            theta (GraphOpParam): Operation parameter.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.
        """
        super().__init__(
            swap=False,
            initialized_modes=[],
            macronode=macronode,
            parameters=[theta],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_MEASUREMENT

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "theta": self.parameters[0],
            "readout": self.readout,
        }


class Initialization(Operation):
    r"""Measure :math:`\hat{x} \sin \theta+\hat{p} \cos \theta` and initialize a mode.

    A mode is initialized with a squeezing angle :math:`\phi=\theta+\frac{\pi}{2}`.
    x-squeezed states correspond to :math:`\phi=0`, while p-squeezed states correspond to :math:`\phi=\pi/2`.
    """

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        theta: GraphOpParam,
        initialized_modes: tuple[int, int],
        *,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            theta (GraphOpParam): Operation parameter.
            initialized_modes (tuple[int, int]): Indices of the initialized modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.
        """
        super().__init__(
            swap=False,
            initialized_modes=list(initialized_modes),
            macronode=macronode,
            parameters=[theta],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_INITIALIZATION

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "theta": self.parameters[0],
            "initialized_modes": tuple(self.initialized_modes),
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class PhaseRotation(Operation):
    r"""Phase rotation gate :math:`R(\phi)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        phi: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            phi (GraphOpParam): Operation parameter.
            swap (bool): Whether to swap the two output modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.
        """
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[phi],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_PHASE_ROTATION

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "phi": self.parameters[0],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class ShearXInvariant(Operation):
    r"""Shear gate (X-invariant) :math:`P(\kappa)`.

    .. math::

        \hat{P}^{\dagger}(\kappa)\binom{\hat{x}}{\hat{p}} \hat{P}(\kappa)=\left(\begin{array}{cc}
        1 & 0 \\
        2 \kappa & 1
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =P(\kappa)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        kappa: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            kappa (GraphOpParam): Operation parameter.
            swap (bool): Whether to swap the two output modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.
        """
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[kappa],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "kappa": self.parameters[0],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class ShearPInvariant(Operation):
    r"""Shear gate (P-invariant) :math:`Q(\eta)`.

    .. math::

        \hat{Q}^{\dagger}(\eta)\binom{\hat{x}}{\hat{p}} \hat{Q}(\eta)=\left(\begin{array}{cc}
        1 & 2 \eta \\
        0 & 1
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =Q(\eta)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        eta: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            eta (GraphOpParam): Operation parameter.
            swap (bool): Whether to swap the two output modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.
        """
        super().__init__(
            macronode=macronode,
            parameters=[eta],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            swap=swap,
            initialized_modes=[],
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "eta": self.parameters[0],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class Squeezing(Operation):
    r"""Squeezing gate with some unusual phase rotation.

    Defined as :math:`R\left(-\frac{\pi}{2}\right) S_{\text{V}}(\cot \theta)`.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}} \\

    .. math::

        \hat{S}_\text{V}^{\dagger}(c)\binom{\hat{x}}{\hat{p}}
        \hat{S}_\text{V}(c)=\left(\begin{array}{cc}
        \frac{1}{c} & 0 \\
        0 & c
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S_\text{V}(c)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        theta: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            theta (GraphOpParam): Operation parameter.
            swap (bool): Whether to swap the two output modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.

        Raises:
            ValueError: If an invalid value for `theta` is provided
        """
        if not isinstance(theta, FeedForward) and equiv_mod_pi(theta, 0):
            msg = "`theta` must not be an integer multiple of pi."
            raise ValueError(msg)
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[theta],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SQUEEZING

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "theta": self.parameters[0],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class Squeezing45(Operation):
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

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        theta: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            theta (GraphOpParam): Operation parameter.
            swap (bool): Whether to swap the two output modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.

        Raises:
            ValueError: If an invalid value for `theta` is provided.
        """
        if not isinstance(theta, FeedForward) and equiv_mod_pi(theta, 0):
            msg = "`theta` must not be an integer multiple of pi."
            raise ValueError(msg)
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[theta],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_SQUEEZING_45

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "theta": self.parameters[0],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class ArbitraryFirst(Operation):
    r"""The first operation for arbitrary gaussian gate :math:`R(\alpha) S(\lambda) R(\beta)`.

    Arbitrary gate is represented with two operations, where
    ``ArbitraryFirst`` is the first and ``ArbitrarySecond`` is the second.

    ``ArbitraryFirst`` and ``ArbitrarySecond`` must be sequentially connected.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}} \\

    .. math::

        \hat{S}^{\dagger}(r)\binom{\hat{x}}{\hat{p}} \hat{S}(r)=\left(\begin{array}{cc}
        e^{-r} & 0 \\
        0 & e^r
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S(r)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        alpha: GraphOpParam,
        beta: GraphOpParam,
        lam: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): The macronode coordinate.
            alpha (GraphOpParam): Operation parameter `alpha`.
            beta (GraphOpParam): Operation parameter `beta`.
            lam (GraphOpParam): Operation parameter `lam`.
            swap (bool): Whether to swap the two output modes of the macronode.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.
        """
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[alpha, beta, lam],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_ARBITRARY_FIRST

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "alpha": self.parameters[0],
            "beta": self.parameters[1],
            "lam": self.parameters[2],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class ArbitrarySecond(Operation):
    r"""The second operation for arbitrary gaussian gate :math:`R(\alpha) S(\lambda) R(\beta)`.

    Arbitrary gate is represented with two operations, where
    ``ArbitraryFirst`` is the first and ``ArbitrarySecond`` is the second.

    ``ArbitraryFirst`` and ``ArbitrarySecond`` must be sequentially connected.

    .. math::

        \hat{R}^{\dagger}(\phi)\binom{\hat{x}}{\hat{p}} \hat{R}(\phi)=\left(\begin{array}{cc}
        \cos \phi & -\sin \phi \\
        \sin \phi & \cos \phi
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =R(\phi)\binom{\hat{x}}{\hat{p}} \\

    .. math::

        \hat{S}^{\dagger}(r)\binom{\hat{x}}{\hat{p}} \hat{S}(r)=\left(\begin{array}{cc}
        e^{-r} & 0 \\
        0 & e^{r}
        \end{array}\right)\binom{\hat{x}}{\hat{p}}
        =S(r)\binom{\hat{x}}{\hat{p}}
    """

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        alpha: GraphOpParam,
        beta: GraphOpParam,
        lam: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): The macronode coordinate.
            alpha (GraphOpParam): Operation parameter `alpha`.
            beta (GraphOpParam): Operation parameter `beta`.
            lam (GraphOpParam): Operation parameter `lam`.
            swap (bool): Whether to swap the two output modes of the macronode.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.
        """
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[alpha, beta, lam],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_ARBITRARY_SECOND

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "alpha": self.parameters[0],
            "beta": self.parameters[1],
            "lam": self.parameters[2],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class ControlledZ(Operation):
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

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        g: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            g (GraphOpParam): Operation parameter.
            swap (bool): Whether to swap the two output modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.
        """
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[g],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_CONTROLLED_Z

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "g": self.parameters[0],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class BeamSplitter(Operation):
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

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        sqrt_r: GraphOpParam,
        theta_rel: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            sqrt_r (GraphOpParam): Operation parameter `sqrt_r`.
            theta_rel (GraphOpParam): Operation parameter `theta_rel`.
            swap (bool): Whether to swap the two output modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.

        Raises:
            ValueError: If an invalid value for `sqrt_r` is provided.
        """
        if not isinstance(sqrt_r, FeedForward) and (sqrt_r < 0 or sqrt_r > 1):
            msg = "`sqrt_r` must be in the range [0, 1]."
            raise ValueError(msg)
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[sqrt_r, theta_rel],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_BEAM_SPLITTER

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "sqrt_r": self.parameters[0],
            "theta_rel": self.parameters[1],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class TwoModeShear(Operation):
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

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        a: GraphOpParam,
        b: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            a (GraphOpParam): Operation parameter `a`.
            b (GraphOpParam): Operation parameter `b`.
            swap (bool): Whether to swap the two output modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.
        """
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[a, b],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "a": self.parameters[0],
            "b": self.parameters[1],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }


class Manual(Operation):
    """Interaction with manually specified four homodyne angles."""

    def __init__(  # noqa: PLR0913
        self,
        macronode: tuple[int, int],
        theta_a: GraphOpParam,
        theta_b: GraphOpParam,
        theta_c: GraphOpParam,
        theta_d: GraphOpParam,
        *,
        swap: bool,
        displacement_k_minus_1: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        displacement_k_minus_n: tuple[GraphOpParam, GraphOpParam] = (0, 0),
        readout: bool = False,
    ) -> None:
        """Initialize the instance by calling the `Operation` class's initializer.

        Args:
            macronode (tuple[int, int]): Macronode coordinate.
            theta_a (GraphOpParam): Operation parameter.
            theta_b (GraphOpParam): Operation parameter.
            theta_c (GraphOpParam): Operation parameter.
            theta_d (GraphOpParam): Operation parameter.
            swap (bool): Whether to swap the two output modes.
            displacement_k_minus_1 (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one above it. Defaults to (0, 0).
            displacement_k_minus_n (tuple[GraphOpParam, GraphOpParam]): Displacement (x, p) applied to the edge
                connecting the current macronode with the one to the left. Defaults to (0, 0).
            readout (bool): Readout the measured values.

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
        super().__init__(
            swap=swap,
            initialized_modes=[],
            macronode=macronode,
            parameters=[theta_a, theta_b, theta_c, theta_d],
            displacement_k_minus_1=displacement_k_minus_1,
            displacement_k_minus_n=displacement_k_minus_n,
            readout=readout,
        )

    def type(self) -> PbOperation.OperationType:  # noqa: D102
        return PbOperation.OPERATION_TYPE_MANUAL

    def _get_init_args(self) -> dict[str, Any]:
        return {
            "macronode": self.macronode,
            "theta_a": self.parameters[0],
            "theta_b": self.parameters[1],
            "theta_c": self.parameters[2],
            "theta_d": self.parameters[3],
            "swap": self.swap,
            "displacement_k_minus_1": self.displacement_k_minus_1,
            "displacement_k_minus_n": self.displacement_k_minus_n,
            "readout": self.readout,
        }
