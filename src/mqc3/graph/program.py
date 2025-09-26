"""Graph representation of continuous variable quantum computing."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from mqc3.feedforward import FeedForward, FeedForwardFunction
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    ArbitraryFirst,
    ArbitrarySecond,
    BeamSplitter,
    ControlledZ,
    GraphOpParam,
    Initialization,
    Manual,
    Measurement,
    ModeMeasuredVariable,
    Operation,
    PhaseRotation,
    PosMeasuredVariable,
    ShearPInvariant,
    ShearXInvariant,
    Squeezing,
    Squeezing45,
    TwoModeShear,
    Wiring,
)
from mqc3.pb.io import ProtoFormat, load, save
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphFF as PbGraphFF
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphOperation as PbOperation
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphRepresentation as PbGraphRepr
from mqc3.utility import OneToOneDict

if TYPE_CHECKING:
    from pathlib import Path

    from mqc3.pb.mqc3_cloud.common.v1.function_pb2 import PythonFunction as PbPythonFunction


def _through(h: int, w: int) -> Operation:
    return Wiring((h, w), swap=False)


class GraphRepr:
    """Graph representation of continuous variable quantum computing.

    Overview of the graph representation:

    ::

                    d           n_{(N-1,0)}   n_{(N-1,1)}   n_{(N-1,M-2)}
                    v             v             v             v
        d       > n_{(0,0)}   > n_{(0,1)}   > n_{(0,2)}   > n_{(0,M-1)}
        d       > n_{(1,0)}   > n_{(1,1)}   > n_{(1,2)}   > n_{(1,M-1)}
        ..      > ..          > ..          > ..          > ..
        d       > n_{(N-1,0)} > n_{(N-1,1)} > ..          > n_{(N-1,M-1)}
                    v             v             v

    * `N` = `n_local_macronodes`
    * `M` = `n_steps`
    * `n` stands for `macronode`
    * `d` stands for default mode
    * We use `h` for the first argument of the coordinates and `w` for the second one.
        * Examples:
        * (0, 1) => h=0, w=1
        * (N-1, M-1) => h=N-1, w=M-1
    """

    def __init__(
        self,
        n_local_macronodes: int,
        n_steps: int,
    ) -> None:
        """Initialize graph representation.

        Args:
            n_local_macronodes (int): The number of local macronodes.
            n_steps (int): The number of steps.

        Raises:
            ValueError: If an input argument has wrong value.

        Examples:
            >>> from mqc3.graph import GraphRepr
            >>> from mqc3.graph.visualize import savefig
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> savefig(graph, "graph.png")
        """
        if n_local_macronodes < 0:
            msg = "`n_local_macronodes` must be a non-negative integer."
            raise ValueError(msg)
        if n_steps < 0:
            msg = "`n_steps` must be a non-negative integer."
            raise ValueError(msg)
        self._n_local_macronodes = n_local_macronodes
        self._n_steps = n_steps
        self._operations: dict[tuple[int, int], Operation] = {}
        for w in range(n_steps):
            for h in range(n_local_macronodes):
                self._operations[h, w] = _through(h, w)

    @property
    def n_local_macronodes(self) -> int:
        """Get the number of local macronodes.

        Returns:
            int: The number of local macronodes.

        Examples:
            >>> from mqc3.graph import GraphRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> graph.n_local_macronodes
            2
        """
        return self._n_local_macronodes

    @property
    def operations(self) -> dict[tuple[int, int], Operation]:
        """Get the operations.

        Returns:
            dict[tuple[int, int], Operation]: The operations.

        Examples:
            >>> from mqc3.graph import GraphRepr, ops
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> op = ops.PhaseRotation((1, 2), phi=1, swap=True)
            >>> graph.place_operation(op)
            >>> graph.operations
            {(0, 0): Wiring(macronode=(0, 0), parameters=[], swap=False, initialized_modes=[], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False), (1, 0): Wiring(macronode=(1, 0), parameters=[], swap=False, initialized_modes=[], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False), (0, 1): Wiring(macronode=(0, 1), parameters=[], swap=False, initialized_modes=[], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False), (1, 1): Wiring(macronode=(1, 1), parameters=[], swap=False, initialized_modes=[], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False), (0, 2): Wiring(macronode=(0, 2), parameters=[], swap=False, initialized_modes=[], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False), (1, 2): PhaseRotation(macronode=(1, 2), parameters=[1], swap=True, initialized_modes=[], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False)}
        """  # noqa: E501
        return self._operations

    @property
    def n_steps(self) -> int:
        """Get the number of steps.

        Returns:
            int: The number of steps.

        Examples:
            >>> from mqc3.graph import GraphRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> graph.n_steps
            3
        """
        return self._n_steps

    @property
    def n_total_macronodes(self) -> int:
        """Get the total number of macronodes.

        The total number of macronodes can be calculated as: `self.n_local_macronodes * self.n_steps`.

        Returns:
            int: The total number of macronodes.

        Examples:
            >>> from mqc3.graph import GraphRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> graph.n_total_macronodes
            6
        """
        return self.n_local_macronodes * self.n_steps

    def get_index(self, h: int, w: int) -> int:
        """Get the index of a coordinate.

        Args:
            h (int): 0 <= `h` < `n_local_macronodes`
            w (int): 0 <= `w` < `n_steps`


        Returns:
            int: The index of (h, w).

        Raises:
            IndexError: The given `h` or `w` is out of range.

        Examples:
            >>> from mqc3.graph import GraphRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> graph.get_index(h=1, w=2)
            5
        """
        if not (h >= 0 and h < self.n_local_macronodes):
            message = "`h` must be in the range [0, self.n_local_macronodes) "
            message += f"({h=}, {self.n_local_macronodes=})."
            raise IndexError(message)
        if not (w >= 0 and w < self.n_steps):
            message = "`w` must be in the range [0, `self.n_steps`) "
            message += f"(w={w}, self.n_steps={self.n_steps})."
            raise IndexError(message)
        return self.n_local_macronodes * w + h

    def get_coord(self, i: int) -> tuple[int, int]:
        """Get the coordinate of an index.

        Args:
            i (int): 0 <= `i` < `n_total_macronodes`.

        Returns:
            tuple[int, int]: The coordinate of `i`-th macronode.

        Raises:
            IndexError: The given `i` is out of range.

        Examples:
            >>> from mqc3.graph import GraphRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> graph.get_coord(i=5)
            (1, 2)
        """
        if i < 0 or i >= self.n_total_macronodes:
            msg = f"`i` must be in the range [0, {self.n_total_macronodes})."
            raise IndexError(msg)

        w, h = divmod(i, self.n_local_macronodes)
        return (h, w)

    def get_operation(self, h: int, w: int) -> Operation:
        """Get the operation which uses macronode (h, w).

        Args:
            h (int): 0 <= `h` < `n_local_macronodes`.
            w (int): 0 <= `w` < `n_steps`.

        Returns:
            Operation: The operation which uses macronode (h, w).

        Examples:
            >>> from mqc3.graph import GraphRepr, ops
            >>> from mqc3.graph import ops
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> op = ops.PhaseRotation((1, 2), phi=1, swap=True)
            >>> graph.place_operation(op)
            >>> graph.get_operation(1, 2)
            PhaseRotation(macronode=(1, 2), parameters=[1], swap=True, initialized_modes=[], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False)
        """  # noqa: E501
        return self._operations[h, w]

    def get_operation_type(self, h: int, w: int) -> PbOperation.OperationType:
        """Get the type of the operation which uses macronode (h, w).

        Args:
            h (int): 0 <= `h` < `n_local_macronodes`
            w (int): 0 <= `w` < `n_steps`

        Returns:
            PbOperation.OperationType: The type of the operation which uses macronode (h, w).
        """
        return self.get_operation(h, w).type()

    def is_measurement(self, h: int, w: int) -> bool:
        """Check if the operation in macronode (h, w) is either measurement or initialization.

        Args:
            h (int): 0 <= `h` < `n_local_macronodes`.
            w (int): 0 <= `w` < `n_steps`.

        Returns:
            bool: Whether the operation in macronode (h, w) is either measurement or initialization.
        """
        operation_type = self.get_operation_type(h, w)
        return operation_type is PbOperation.OPERATION_TYPE_MEASUREMENT

    def place_operation(self, op: Operation) -> None:
        """Place the operation.

        Args:
            op (Operation): The operation to place.

        Raises:
            ValueError: The macronode position is out of range.

        Examples:
            >>> from mqc3.graph import GraphRepr, ops
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> op = ops.PhaseRotation((1, 2), phi=1, swap=True)
            >>> graph.place_operation(op)
            >>> graph.get_operation(1, 2)
            PhaseRotation(macronode=(1, 2), parameters=[1], swap=True, initialized_modes=[], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False)
        """  # noqa: E501
        h, w = op.macronode
        if h >= self.n_local_macronodes:
            msg = f"Invalid macronode position: {h} >= n_local_macronodes ({self.n_local_macronodes})."
            raise ValueError(msg)
        if w >= self.n_steps:
            msg = f"Invalid macronode position: {w} >= n_steps ({self.n_steps})."
            raise ValueError(msg)
        self._operations[op.macronode] = op

    def get_measured_value(self, h: int, w: int, bd: int) -> PosMeasuredVariable:
        """Get the measured value of macronode (h, w).

        Args:
            h (int): 0 <= `h` < `n_local_macronodes`.
            w (int): 0 <= `w` < `n_steps`.
            bd (int): 0 or 1 (b or d).

        Returns:
            PosMeasuredVariable: The measured value of macronode (h, w).
        """
        return PosMeasuredVariable(h, w, bd)

    def get_mode_measured_value(self, mode: int) -> ModeMeasuredVariable:
        """Get the measured value of the mode.

        Args:
            mode (int): The mode number.

        Returns:
            PosMeasuredVariable: The measured value of the mode.
        """
        return ModeMeasuredVariable(mode)

    def get_readout_macronode_indices(self) -> list[int]:
        """Get the indices of readout macronodes.

        Returns:
            list[int]: The indices of readout macronodes, sorted in ascending order.
        """
        ret: list[int] = []
        for c, op in self._operations.items():
            if op.readout:
                ret.append(self.get_index(*c))
        return sorted(ret)

    def reduce_steps(self, n_steps: int) -> None:
        """Remove redundant steps to reduce the number of steps in the graph representation to `n_steps`.

        The macronodes in the steps to be removed during the reduction process must all be `Wiring` operations.

        Args:
            n_steps (int): The number of steps, which must be smaller than the current `n_steps`.

        Raises:
            ValueError: The given `n_steps` is larger than the current one.
            TypeError: If a step being removed contains an operation other than `Wiring`.

        Examples:
            >>> from mqc3.graph import GraphRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> graph.n_steps
            3
            >>> graph.reduce_steps(2)
            >>> graph.n_steps
            2
        """
        if n_steps > self.n_steps:
            msg = f"`n_steps` ({n_steps}) is greater than the current value ({self.n_steps})."
            raise ValueError(msg)

        old_n_steps = self.n_steps
        self._n_steps = n_steps
        for w in range(n_steps, old_n_steps):
            for h in range(self.n_local_macronodes):
                if not isinstance(self.get_operation(h, w), Wiring):
                    raise TypeError
                self._operations.pop((h, w))

    def increase_local_macronodes(self, new_n_local_macronodes: int) -> None:
        """Increase the local macronodes.

        Args:
            new_n_local_macronodes (int): The new number of local macronodes.

        Raises:
            ValueError: The new number of local macronodes is smaller than the current one.

        Examples:
            >>> from mqc3.graph import GraphRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> graph.n_local_macronodes
            2
            >>> graph.increase_local_macronodes(3)
            >>> graph.n_local_macronodes
            3
        """
        if new_n_local_macronodes < self.n_local_macronodes:
            msg = f"`new_n_local_macronodes` ({new_n_local_macronodes}) is smaller than "
            msg += f"the current value ({self.n_local_macronodes})."
            raise ValueError(msg)
        old_n_local_macronodes = self.n_local_macronodes

        # Increase the number of macronodes
        self._n_local_macronodes = new_n_local_macronodes
        # Add through gates
        for w in range(self.n_steps):
            for h in range(old_n_local_macronodes, new_n_local_macronodes):
                self._operations[h, w] = _through(h, w)

    def is_swap_macronode(self, h: int, w: int) -> bool:
        """Check if the macronode (h, w) is either swap or through.

        Args:
            h (int): 0 <= `h` < `n_local_macronodes`.
            w (int): 0 <= `w` < `n_steps`.

        Returns:
            bool: Whether the macronode (h, w) is either swap or through.

        Examples:
            >>> from mqc3.graph import GraphRepr, ops
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> op = ops.PhaseRotation((1, 2), swap=True, phi=1)
            >>> graph.place_operation(op)
            >>> graph.is_swap_macronode(1, 1)
            False
            >>> graph.is_swap_macronode(1, 2)
            True
        """
        op = self.get_operation(h=h, w=w)
        return op.swap

    def calc_io_of_macronode(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Calculate input/output modes of macronode at (`h`, `w`).

        Args:
            h (int): The first argument of the coordinates.
            w (int): The second argument of the coordinates.

        Returns:
            tuple[int, int, int, int]: Input and output modes of macronode:
                [left mode index, up mode index, right mode index, down mode index].

        Examples:
            >>> from mqc3.graph import GraphRepr, ops
            >>> from mqc3.graph.constant import BLANK_MODE
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> op = ops.Initialization((1, 2), initialized_modes=[0, BLANK_MODE], theta=1)
            >>> graph.place_operation(op)
            >>> graph.calc_io_of_macronode(1, 2)
            (-1, -1, -1, 0)
        """
        # First N elements are input modes from left.
        # The last element is an input mode from up.
        # N = self.n_local_macronodes
        modes = [BLANK_MODE] * (self.n_local_macronodes + 1)

        left = BLANK_MODE
        up = BLANK_MODE
        for j in range(w + 1):
            nh = h + 1 if j == w else self.n_local_macronodes
            for i in range(nh):
                left = modes[i]
                up = modes[-1]

                op = self.get_operation(i, j)
                if op.type() == PbOperation.OPERATION_TYPE_MEASUREMENT:
                    modes[i] = modes[-1] = BLANK_MODE
                elif op.type() == PbOperation.OPERATION_TYPE_INITIALIZATION:
                    modes[-1] = op.initialized_modes[0]
                    modes[i] = op.initialized_modes[1]
                elif self.is_swap_macronode(i, j):
                    modes[-1], modes[i] = left, up

        return (left, up, modes[h], modes[-1])

    def io_modes_dict(self) -> dict[tuple[int, int], tuple[int, int, int, int]]:
        """Calculate input/output modes of all macronodes and generate dict.

        Returns:
            dict[tuple[int, int], tuple[int, int, int, int]]:
                Dict from coordinate of macronode:
                    (h, w)
                to input and output modes of macronode:
                    (left mode index, up mode index, right mode index, down mode index).

        Examples:
            >>> from mqc3.graph import GraphRepr, ops
            >>> from mqc3.graph.constant import BLANK_MODE
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> op = ops.Initialization((1, 2), initialized_modes=[0, BLANK_MODE], theta=1)
            >>> graph.place_operation(op)
            >>> graph.io_modes_dict()[(1, 2)]
            (-1, -1, -1, 0)
        """
        # First `n_local_macronodes` elements are input modes from left.
        # The last element is an input mode from up.
        modes = [BLANK_MODE] * (self.n_local_macronodes + 1)

        io_modes_dict = {}
        for w in range(self.n_steps):
            for h in range(self.n_local_macronodes):
                left, up = modes[h], modes[-1]
                down, right = up, left
                op = self.get_operation(h, w)
                if op.type() == PbOperation.OPERATION_TYPE_MEASUREMENT:
                    down, right = BLANK_MODE, BLANK_MODE
                elif op.type() == PbOperation.OPERATION_TYPE_INITIALIZATION:
                    down, right = op.initialized_modes
                elif self.is_swap_macronode(h, w):
                    down, right = left, up

                io_modes_dict[h, w] = (left, up, right, down)

                # Update the left mode at `h` to the current right mode
                modes[h] = io_modes_dict[h, w][2]
                # Update the up mode to the current down mode
                modes[-1] = io_modes_dict[h, w][3]

        return io_modes_dict

    def calc_mode_operations(self, mode: int) -> list[Operation]:  # noqa: C901
        """Construct a list of operations on `mode`.

        Args:
            mode (int): Target mode number.

        Returns:
            list[Operation]: A list of operations.

        Raises:
            ValueError: The input mode is blank mode or not found.

        Examples:
            >>> from mqc3.graph import GraphRepr, ops
            >>> from mqc3.graph.constant import BLANK_MODE
            >>> BLANK_MODE
            -1
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> op = ops.Initialization((0, 0), initialized_modes=[0, BLANK_MODE], theta=1)
            >>> graph.place_operation(op)
            >>> op = ops.PhaseRotation((0, 1), swap=True, phi=1)
            >>> graph.place_operation(op)
            >>> for op in graph.calc_mode_operations(0):
            ...     print(op)
            Initialization(macronode=(0, 0), parameters=[1], swap=False, initialized_modes=[0, -1], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False)
            PhaseRotation(macronode=(0, 1), parameters=[1], swap=True, initialized_modes=[], displacement_k_minus_1=(0, 0), displacement_k_minus_n=(0, 0), readout=False)
        """  # noqa: E501
        if mode == BLANK_MODE:
            msg = "The target mode index must not be blank mode."
            raise ValueError(msg)

        # Find the starting position
        h, w = (-1, 0)
        from_left = True
        for i in range(self.n_total_macronodes):
            cur_h, cur_w = self.get_coord(i)
            op = self.get_operation(h=cur_h, w=cur_w)
            if op.type() == PbOperation.OPERATION_TYPE_INITIALIZATION:
                if op.initialized_modes[0] == mode:
                    from_left = False
                    h, w = cur_h, cur_w
                    break
                if op.initialized_modes[1] == mode:
                    from_left = True
                    h, w = cur_h, cur_w
                    break

        if h == -1:
            msg = f"Mode {mode} is not found."
            raise ValueError(msg)

        mode_operations = []
        while w < self.n_steps:
            op = self.get_operation(h=h, w=w)
            # Skip Wiring and the second node of Arbitrary
            if op.type() != PbOperation.OPERATION_TYPE_WIRING:
                mode_operations.append(op)
            if op.type() == PbOperation.OPERATION_TYPE_MEASUREMENT:
                break

            # Move on to the next node
            swap = self.is_swap_macronode(h, w)
            if (swap and from_left) or (not swap and not from_left):
                # Down
                from_left = False
                h, w = (h + 1, w) if h + 1 < self.n_local_macronodes else (0, w + 1)
            elif (swap and not from_left) or (not swap and from_left):
                # Right
                from_left = True
                w += 1

        return mode_operations

    def proto(self) -> PbGraphRepr:  # noqa: D102
        operations = []
        functions = []
        nlffs = []
        for op in self._operations.values():
            operations.append(construct_proto_from_graph_operation(op, self))

            to_ind = self.get_index(*op.macronode)
            for ind, p in enumerate(op.parameters):
                if not isinstance(p, FeedForward):
                    continue

                nlff, ff_proto = _construct_nlff_proto_from_ff(p, to_ind, ind, len(functions), self)

                functions.append(ff_proto)
                nlffs.append(nlff)

            for ind, disp in zip(
                _DispInd,
                [
                    op.displacement_k_minus_1[0],
                    op.displacement_k_minus_1[1],
                    op.displacement_k_minus_n[0],
                    op.displacement_k_minus_n[1],
                ],
                strict=True,
            ):
                if not isinstance(disp, FeedForward):
                    continue

                nlff, ff_proto = _construct_nlff_proto_from_ff(disp, to_ind, ind, len(functions), self)

                functions.append(ff_proto)
                nlffs.append(nlff)

        return PbGraphRepr(
            n_local_macronodes=self.n_local_macronodes,
            n_steps=self.n_steps,
            operations=operations,
            nlffs=nlffs,
            functions=functions,
        )

    @staticmethod
    def construct_from_proto(proto: PbGraphRepr) -> GraphRepr:  # noqa: D102
        graph = GraphRepr(
            n_local_macronodes=proto.n_local_macronodes,
            n_steps=proto.n_steps,
        )
        for proto_operation in proto.operations:
            op = construct_graph_operation_from_proto(proto_operation, graph.n_local_macronodes)
            graph.place_operation(op)
        for proto_nlff in proto.nlffs:
            h, w = graph.get_coord(proto_nlff.from_macronode)
            v = graph.get_measured_value(h, w, proto_nlff.from_bd)
            f = FeedForwardFunction.construct_from_proto(proto.functions[proto_nlff.function])

            ff = f(v)

            op = graph.get_operation(*graph.get_coord(proto_nlff.to_macronode))
            if proto_nlff.to_parameter == _DispInd.KP1_X:
                op.displacement_k_minus_1 = (ff, op.displacement_k_minus_1[1])
            elif proto_nlff.to_parameter == _DispInd.KP1_P:
                op.displacement_k_minus_1 = (op.displacement_k_minus_1[0], ff)
            elif proto_nlff.to_parameter == _DispInd.KPN_X:
                op.displacement_k_minus_n = (ff, op.displacement_k_minus_n[1])
            elif proto_nlff.to_parameter == _DispInd.KPN_P:
                op.displacement_k_minus_n = (op.displacement_k_minus_n[0], ff)
            else:
                op.parameters[proto_nlff.to_parameter] = ff

        return graph

    def save(self, path: str | Path, proto_format: ProtoFormat = "text") -> None:  # noqa: D102
        save(self.proto(), path, proto_format)

    @staticmethod
    def load(path: str | Path, proto_format: ProtoFormat = "text") -> GraphRepr:  # noqa: D102
        return GraphRepr.construct_from_proto(load(PbGraphRepr, path, proto_format))


def to_pos_measured_variable(var: ModeMeasuredVariable, graph: GraphRepr) -> PosMeasuredVariable:
    """Convert the mode measured variable to the position measured variable.

    Args:
        var (ModeMeasuredVariable): The variable.
        graph (GraphRepr): The graph representation.

    Returns:
        PosMeasuredVariable: The position measured variable.

    Raises:
        ValueError: If the operation for the mode is not found.
    """
    op_list = graph.calc_mode_operations(var.mode)
    if not op_list:
        msg = f"No operation found for mode {var.mode}."
        raise ValueError(msg)

    op = op_list[-1]
    h, w = op.macronode

    left, up, _, _ = graph.calc_io_of_macronode(h, w)
    if var.mode == up:
        return PosMeasuredVariable(h, w, 0)
    if var.mode == left:
        return PosMeasuredVariable(h, w, 1)

    msg = f"No measurement found for mode {var.mode}."
    raise ValueError(msg)


corr_gop_pb = OneToOneDict[PbOperation.OperationType, type[Operation]](
    [
        (PbOperation.OPERATION_TYPE_MEASUREMENT, Measurement),
        (PbOperation.OPERATION_TYPE_INITIALIZATION, Initialization),
        (PbOperation.OPERATION_TYPE_PHASE_ROTATION, PhaseRotation),
        (PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT, ShearXInvariant),
        (PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT, ShearPInvariant),
        (PbOperation.OPERATION_TYPE_SQUEEZING, Squeezing),
        (PbOperation.OPERATION_TYPE_SQUEEZING_45, Squeezing45),
        (PbOperation.OPERATION_TYPE_ARBITRARY_FIRST, ArbitraryFirst),
        (PbOperation.OPERATION_TYPE_ARBITRARY_SECOND, ArbitrarySecond),
        (PbOperation.OPERATION_TYPE_CONTROLLED_Z, ControlledZ),
        (PbOperation.OPERATION_TYPE_BEAM_SPLITTER, BeamSplitter),
        (PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR, TwoModeShear),
        (PbOperation.OPERATION_TYPE_MANUAL, Manual),
        (PbOperation.OPERATION_TYPE_WIRING, Wiring),
    ],
)


class _DispInd(IntEnum):
    KP1_X = 4
    KP1_P = 5
    KPN_X = 6
    KPN_P = 7


def construct_proto_from_graph_operation(op: Operation, graph: GraphRepr) -> PbOperation:
    """Get the proto of the graph operation.

    Returns:
        PbOperation: Proto.
    """

    def default_if_ff(v: GraphOpParam, default: float) -> float:
        return v if not isinstance(v, FeedForward) else default

    return PbOperation(
        type=corr_gop_pb.get_k(type(op)),
        swap=op.swap,
        initialized_modes=op.initialized_modes,
        displacement_k_minus_1=PbOperation.Displacement(
            x=default_if_ff(op.displacement_k_minus_1[0], 0),
            p=default_if_ff(op.displacement_k_minus_1[1], 0),
        ),
        displacement_k_minus_n=PbOperation.Displacement(
            x=default_if_ff(op.displacement_k_minus_n[0], 0),
            p=default_if_ff(op.displacement_k_minus_n[1], 0),
        ),
        macronode=graph.get_index(*op.macronode),
        parameters=(default_if_ff(p, 0) for p in op.parameters),
        readout=op.readout,
    )


def construct_graph_operation_from_proto(proto: PbOperation, n_local_macronodes: int) -> Operation:
    """Construct the graph operation from the proto.

    Args:
        proto (PbOperation): Proto.
        n_local_macronodes (int): The number of local macronodes.

    Returns:
        GraphOperation: The graph operation.
    """
    op_type = corr_gop_pb.get_v(proto.type)
    if op_type is Wiring:
        return op_type(
            (proto.macronode % n_local_macronodes, proto.macronode // n_local_macronodes),
            swap=proto.swap,
            readout=proto.readout,
        )
    if op_type is Initialization:
        return op_type(
            (proto.macronode % n_local_macronodes, proto.macronode // n_local_macronodes),
            proto.parameters[0],
            initialized_modes=(proto.initialized_modes[0], proto.initialized_modes[1]),
            displacement_k_minus_1=(proto.displacement_k_minus_1.x, proto.displacement_k_minus_1.p),
            displacement_k_minus_n=(proto.displacement_k_minus_n.x, proto.displacement_k_minus_n.p),
            readout=proto.readout,
        )
    if op_type is Measurement:
        return op_type(
            (proto.macronode % n_local_macronodes, proto.macronode // n_local_macronodes),
            proto.parameters[0],
            readout=proto.readout,
        )

    return op_type(
        (proto.macronode % n_local_macronodes, proto.macronode // n_local_macronodes),
        *proto.parameters,  # type: ignore  # noqa: PGH003
        swap=proto.swap,
        displacement_k_minus_1=(proto.displacement_k_minus_1.x, proto.displacement_k_minus_1.p),
        displacement_k_minus_n=(proto.displacement_k_minus_n.x, proto.displacement_k_minus_n.p),
    )


def _construct_nlff_proto_from_ff(
    ff: FeedForward,
    to_macronode: int,
    to_parameter: int,
    function: int,
    graph: GraphRepr,
) -> tuple[PbGraphFF, PbPythonFunction]:
    ff_proto = ff.func.proto()

    v = ff.variable
    if isinstance(v, ModeMeasuredVariable):
        v = to_pos_measured_variable(v, graph)

    h, w, bd = v.get_from_operation()
    from_ind = graph.get_index(h, w)

    nlff = PbGraphFF(
        from_macronode=from_ind,
        from_bd=bd,
        to_macronode=to_macronode,
        to_parameter=to_parameter,
        function=function,
    )

    return nlff, ff_proto
