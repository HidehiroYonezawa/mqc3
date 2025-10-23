"""Circuit representation of continuous variable quantum computing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import TYPE_CHECKING

from mqc3.circuit.ops import intrinsic
from mqc3.circuit.ops._base import (
    CircOpParam,
    MeasuredVariable,  # noqa: F401
    Operand,
    Operation,
    QuMode,
)
from mqc3.circuit.state import (
    BosonicState,
    HardwareConstrainedSqueezedState,
    InitialState,
    construct_initial_state_from_proto,
    construct_proto_from_initial_state,
)
from mqc3.feedforward import FeedForward, FeedForwardFunction
from mqc3.pb.io import ProtoFormat, load, save
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitFF as PbCircuitFF
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitRepresentation as PbCircuitRepr
from mqc3.utility import OneToOneDict

if TYPE_CHECKING:
    from pathlib import Path


class CircuitRepr:
    """Circuit representation of continuous variable quantum computing.

    initial_states is used only when simulating the circuit with the :class:`~mqc3.client.SimulatorClient`.
    We assume that each initial state is a hardware constrained squeezed state or
    a superposition of gaussian states with following constraints.

    Constraints

    - The number of modes of each gaussian state is 1.
    """

    def __init__(self, name: str) -> None:
        """Construct a circuit representation.

        Args:
            name (str): Circuit name.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> circuit = CircuitRepr("example")
            >>> circuit.name
            'example'
        """
        self._name: str = name
        self._n_modes: int = 0
        self._ops: list[Operation] = []
        self._initial_states: list[InitialState] = []

    def _insert(self, op: Operation) -> None:
        self._ops.append(op)

    def _push_one(self, modes: list[int], one: int) -> None:
        try:
            modes.append(int(one))
        except ValueError as err:
            message = "The argument of Q() must be int or list[int]."
            raise RuntimeError(message) from err

    def _push(self, modes: list[int], arg: int | list[int]) -> None:
        if isinstance(arg, Iterable):
            for one in arg:
                self._push_one(modes, one)
        else:
            self._push_one(modes, arg)

    @property
    def name(self) -> str:
        """Get the name of the circuit representation.

        Returns:
            str: Name of the circuit representation.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> circuit = CircuitRepr("example")
            >>> circuit.name
            'example'
        """
        return self._name

    @property
    def n_modes(self) -> int:
        """Get the number of modes in the circuit.

        Returns:
            int: The number of modes.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> circuit = CircuitRepr("example")
            >>> circuit.n_modes
            0
            >>> circuit.Q(0)
            [QuMode(id=0)]
            >>> circuit.n_modes
            1
        """
        return self._n_modes

    @property
    def n_operations(self) -> int:
        """Get the number of operations in the circuit.

        Returns:
            int: The number of operations.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> circuit = CircuitRepr("example")
            >>> circuit.n_operations
            0
            >>> circuit.Q(0) | intrinsic.Displacement(0, 1)
            [QuMode(id=0)]
            >>> circuit.n_operations
            1
        """
        return len(self._ops)

    def Q(self, *args: int | list[int]) -> Operand:
        """Create an operand.

        Args:
            args: `int` or iterable of `int`.

        Returns:
            Operand: Operand.

        Note:
            When adding an operand which contains `i`, modes with ids 0 to `i-1` are added implicitly.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0)
            [QuMode(id=0)]
            >>> circuit.Q(0, 1)
            [QuMode(id=0), QuMode(id=1)]
            >>> circuit.Q([1, 2])
            [QuMode(id=1), QuMode(id=2)]
        """
        modes = []
        for arg in args:
            self._push(modes, arg)
        self._n_modes = max(self.n_modes, max(modes) + 1)
        while len(self._initial_states) < self._n_modes:
            self._initial_states.append(HardwareConstrainedSqueezedState())
        return Operand([QuMode(mode) for mode in modes], self)

    def get_operation(self, i: int) -> Operation:
        """Get the operation.

        Args:
            i (int): Index of the operation.

        Returns:
            Operation: Operation.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0, 1)
            [QuMode(id=0)]
            >>> op = circuit.get_operation(0)
            >>> op.name()
            'intrinsic.displacement'
        """
        return self._ops[i]

    def sum_n_macronodes(self) -> int:
        """The total number of macronodes for operations.

        This value is the lower limit of the number of macronodes required for
        converting to graph representation.

        Returns:
            int: The total number of macronodes.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Arbitrary(0, 1, 2)
            [QuMode(id=0)]
            >>> circuit.sum_n_macronodes()
            2
        """
        return sum(op.n_macronodes() for op in self)

    def convert_std_ops_to_intrinsic(self) -> None:
        """Convert all std operations to intrinsic operations.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import std
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0, 1) | std.BeamSplitter(0.0, 1.0)
            [QuMode(id=0), QuMode(id=1)]
            >>> circuit.convert_std_ops_to_intrinsic()
            >>> print(circuit)
            [intrinsic.phase_rotation] [-2.141592653589793] [QuMode(id=0)]
            [intrinsic.phase_rotation] [-1.5707963267948966] [QuMode(id=1)]
            [intrinsic.manual] [0, 1.5707963267948966, 0.0, 1.5707963267948966] [QuMode(id=0), QuMode(id=1)]
            [intrinsic.phase_rotation] [2.141592653589793] [QuMode(id=0)]
            [intrinsic.phase_rotation] [1.5707963267948966] [QuMode(id=1)]
        """
        original_ops = deepcopy(self._ops)
        self._ops.clear()
        for op in original_ops:
            for intrinsic_op in op.to_intrinsic_ops():
                self._insert(intrinsic_op)

    @property
    def initial_states(self) -> list[InitialState]:
        """Get list of initial states."""
        return self._initial_states

    def set_initial_state(self, index: int, state: InitialState) -> None:
        """Set a InitialState at the specified mode.

        If state is instance of BosonicState:
        The given state must be a single mode.
        The index must correspond to an already created mode.

        Args:
            index (int): The index of the mode.
            state (InitialState): The InitialState to be set.

        Raises:
            ValueError: If one of the Following two cases occurs.

                1. If the index does not correspond to an already created mode.
                2. If the state is BosonicState instance and is not a BosonicState constructed from a single mode.


        Examples:
            >>> import numpy as np
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> from mqc3.circuit.state import GaussianState, BosonicState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0, 1)
            [QuMode(id=0)]
            >>> coeffs = np.array([1.+0.j], dtype=np.complex128)
            >>> mean = np.array([1.+0.j, 0.+0.j], dtype=np.complex128)
            >>> cov = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
            >>> new_state = BosonicState(coeffs, [GaussianState(mean, cov)])
            >>> circuit.set_initial_state(0, new_state)
            >>> print(circuit.get_initial_state(0))
            BosonicState(n_modes=1)
        """
        if not 0 <= index < self.n_modes:
            msg = f"Invalid index {index}. It must refer to a previously created mode."
            raise ValueError(msg)

        # Check if state is a BosonicState whose n_modes is 1.
        if isinstance(state, BosonicState) and state.n_modes != 1:
            msg = "The state argument must be a BosonicState with a single mode."
            raise ValueError(msg)

        self._initial_states[index] = state

    def get_initial_state(self, index: int) -> InitialState:
        """Get the initial state at the specified index.

        The given index must correspond to an already created mode.

        Args:
            index(int): The index of the mode.

        Returns:
            InitialState: The initial state at the specified mode.

        Raises:
            ValueError: If the index does not correspond to an already created mode.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> from mqc3.circuit.state import BosonicState
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0, 1)
            [QuMode(id=0)]
            >>> circuit.set_initial_state(0, BosonicState.vacuum())
            >>> print(circuit.get_initial_state(0))
            BosonicState(n_modes=1)
        """
        if not 0 <= index < self.n_modes:
            msg = f"Invalid index {index}. The index must be that of an already created mode."
            raise ValueError(msg)

        return self._initial_states[index]

    def __len__(self) -> int:
        """Return the number of operations in the circuit.

        Returns:
            int: The number of operations.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0, 1)
            [QuMode(id=0)]
            >>> len(circuit)
            1
        """
        return len(self._ops)

    def __iter__(self) -> Iterator[Operation]:
        """Iterate over operations in the circuit.

        Yields:
            Operation: Operation.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0, 1)
            [QuMode(id=0)]
            >>> circuit.Q(1) | intrinsic.PhaseRotation(1.0)
            >>> for op in circuit:
            ...     print(op)
            [intrinsic.displacement] [0, 1] [QuMode(id=0)]
            [intrinsic.phase_rotation] [1.0] [QuMode(id=1)]
        """
        yield from self._ops

    def __str__(self) -> str:
        """Get the operation list as a string.

        Returns:
            str: Operation list.

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0, 1)
            [QuMode(id=0)]
            >>> circuit.Q(1) | intrinsic.PhaseRotation(1.0)
            >>> print(circuit)
            [intrinsic.displacement] [0.0, 1.0]
            [intrinsic.phase_rotation] [1.0]
        """
        ret = ""
        for op in self._ops:
            ret += str(op) + "\n"
        return ret.rstrip()

    def __repr__(self) -> str:
        r"""Get the string representation of the circuit.

        Returns:
            str: String representation

        Examples:
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.Displacement(0, 1)
            [QuMode(id=0)]
            >>> circuit.Q(1) | intrinsic.PhaseRotation(1.0)
            [QuMode(id=1)]
            >>> repr(circuit)
            '[intrinsic.displacement] [0.0, 1.0]\\n[intrinsic.phase_rotation] [1.0]'
        """
        return str(self)

    def proto(self) -> PbCircuitRepr:  # noqa: D102
        ops = []
        nlffs = []
        functions = []
        self._ops = [tmp for op in self._ops for tmp in op.to_intrinsic_ops()]
        for op_ind, op in enumerate(self):
            ops += op.proto()
            for param_ind, param in enumerate(op.parameters()):
                if isinstance(param, FeedForward):
                    from_operation = param.variable.get_from_operation()
                    from_ind = self._ops.index(from_operation)
                    pb_func = param.func.proto()
                    functions.append(pb_func)

                    nlff = PbCircuitFF(
                        function=len(functions) - 1,
                        from_operation=from_ind,
                        to_operation=op_ind,
                        to_parameter=param_ind,
                    )
                    nlffs.append(nlff)

        initial_states = [construct_proto_from_initial_state(initial_state) for initial_state in self._initial_states]

        return PbCircuitRepr(
            n_modes=self.n_modes,
            initial_states=initial_states,
            operations=ops,
            nlffs=nlffs,
            functions=functions,
        )

    @staticmethod
    def construct_from_proto(proto: PbCircuitRepr) -> CircuitRepr:  # noqa: D102
        circuit_repr = CircuitRepr(proto.name)

        nlff_dict: dict[int, list[PbCircuitFF]] = defaultdict(list)
        for pb_nlff in proto.nlffs:
            nlff_dict[pb_nlff.to_operation].append(pb_nlff)

        var_dict = {}
        for op_ind, pb_op in enumerate(proto.operations):
            modes = pb_op.modes
            params: list[float | FeedForward] = list(pb_op.parameters)
            for pb_nlff in nlff_dict[op_ind]:
                var = var_dict[pb_nlff.from_operation]
                pb_func = proto.functions[pb_nlff.function]
                ff = FeedForwardFunction.construct_from_proto(pb_func)
                params[pb_nlff.to_parameter] = ff(var)

            op = construct_operation_from_proto(pb_op.type, params)
            if isinstance(op, intrinsic.Measurement):
                x = circuit_repr.Q(*modes) | op
                var_dict[op_ind] = x
            else:
                circuit_repr.Q(*modes) | op  # pyright: ignore[reportUnusedExpression]

        # Apply initial state
        for i, pb_initial_state in enumerate(proto.initial_states):
            state = construct_initial_state_from_proto(pb_initial_state)
            if state is not None:
                circuit_repr.set_initial_state(i, state)

        return circuit_repr

    def save(self, path: str | Path, proto_format: ProtoFormat = "text") -> None:  # noqa: D102
        save(self.proto(), path, proto_format)

    @staticmethod
    def load(path: str | Path, proto_format: ProtoFormat = "text") -> CircuitRepr:  # noqa: D102
        return CircuitRepr.construct_from_proto(load(PbCircuitRepr, path, proto_format))


corr_cop_pb = OneToOneDict[PbOperation.OperationType, type[intrinsic.Intrinsic]](
    [
        (PbOperation.OPERATION_TYPE_MEASUREMENT, intrinsic.Measurement),
        (PbOperation.OPERATION_TYPE_DISPLACEMENT, intrinsic.Displacement),
        (PbOperation.OPERATION_TYPE_PHASE_ROTATION, intrinsic.PhaseRotation),
        (PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT, intrinsic.ShearPInvariant),
        (PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT, intrinsic.ShearXInvariant),
        (PbOperation.OPERATION_TYPE_SQUEEZING, intrinsic.Squeezing),
        (PbOperation.OPERATION_TYPE_SQUEEZING_45, intrinsic.Squeezing45),
        (PbOperation.OPERATION_TYPE_ARBITRARY, intrinsic.Arbitrary),
        (PbOperation.OPERATION_TYPE_CONTROLLED_Z, intrinsic.ControlledZ),
        (PbOperation.OPERATION_TYPE_BEAM_SPLITTER, intrinsic.BeamSplitter),
        (PbOperation.OPERATION_TYPE_MANUAL, intrinsic.Manual),
        (PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR, intrinsic.TwoModeShear),
    ],
)


def construct_operation_from_proto(
    proto_operation_type: PbOperation.OperationType,
    params: list[CircOpParam],
) -> Operation:
    """Convert from proto format to obtain the Operation object.

    Args:
        proto_operation_type (PbOperation.OperationType): Operation type in proto format.
        params (list[CircOpParam]): Parameters of the operation.

    Returns:
        Operation: Operation object
    """
    op = corr_cop_pb.get_v(proto_operation_type)
    return op(*params)
