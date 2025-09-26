"""Machinery representation of continuous variable quantum computing."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Literal

import mqc3.graph.ops as gops
from mqc3.feedforward import FeedForward, FeedForwardFunction
from mqc3.graph.constant import BLANK_MODE
from mqc3.machinery.macronode_angle import (
    MachineOpParam,
    MacronodeAngle,
    MeasuredVariable,
    convert_gm_param,
    from_graph_operation,
)
from mqc3.machinery.utility import calculate_ff_matrix_kp1, calculate_ff_matrix_kpn, construct_empty_ff_matrix
from mqc3.pb.io import ProtoFormat, load, save
from mqc3.pb.mqc3_cloud.program.v1.machinery_pb2 import (
    FeedForwardCoefficientGenerationMethod as PbFeedForwardCoefficientGenerationMethod,
)
from mqc3.pb.mqc3_cloud.program.v1.machinery_pb2 import MachineryFF as PbMachineryFF
from mqc3.pb.mqc3_cloud.program.v1.machinery_pb2 import MachineryRepresentation as PbMachineryRepr
from mqc3.utility import OneToOneDict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    import numpy as np

    from mqc3.graph.program import GraphRepr
    from mqc3.pb.mqc3_cloud.common.v1.function_pb2 import PythonFunction as PbPythonFunction


class FFCoeffMatrixGenerationMethods(Enum):
    """Generation methods for feedforward coefficient matrix."""

    ZERO_FILLED = 0
    """Zero-filled feedforward coefficient matrix."""

    FROM_MACRONODE_ANGLE = 1
    """Calculate feedforward coefficient matrix from homodyne angles."""


class _FFCoeffMatrixList:
    generator: Callable[[MacronodeAngle, MacronodeAngle], np.ndarray]
    """Generator function for feedforward coefficient matrix."""

    generation_methods: list[FFCoeffMatrixGenerationMethods]
    """Generation methods for feedforward coefficient matrix.
    The length of this list is equal to the number of homodyne angles (``MachineryRepr.n_total_macronodes``)."""

    next_index_distance: int
    """Distance between the indices of the homodyne angles used to generate feedforward coefficient matrix."""

    homodyne_angles: list[MacronodeAngle]
    """Homodyne angles used to generate feedforward coefficient matrix."""

    def __init__(
        self,
        generator: Callable[[MacronodeAngle, MacronodeAngle], np.ndarray],
        next_index_distance: int,
        homodyne_angles: list[MacronodeAngle],
    ) -> None:
        self.generator = generator
        self.next_index_distance = next_index_distance
        self.homodyne_angles = homodyne_angles

        self.generation_methods = [FFCoeffMatrixGenerationMethods.FROM_MACRONODE_ANGLE] * len(homodyne_angles)

    def __getitem__(self, index: int) -> np.ndarray:
        """Get feedforward coefficient matrix.

        Note:
            If the ``index`` is out of range, the method raises an IndexError.

        Args:
            index (int): Index of the homodyne angle (0 <= index < len(self)).

        Returns:
            np.ndarray: Feedforward coefficient matrix.

        Raises:
            ValueError: If the generation method is invalid.
        """
        method = self.generation_methods[index]
        if method == FFCoeffMatrixGenerationMethods.ZERO_FILLED:
            return construct_empty_ff_matrix()
        if method == FFCoeffMatrixGenerationMethods.FROM_MACRONODE_ANGLE:
            angle = self.homodyne_angles[index]

            if index + self.next_index_distance >= len(self.homodyne_angles):
                next_angle = MacronodeAngle(0, 0, 0, 0)
            else:
                next_angle = self.homodyne_angles[index + self.next_index_distance]

            return self.generator(angle, next_angle)

        msg = "Invalid generation method."
        raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.homodyne_angles)

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]


class _FFPosition(IntEnum):
    THETA_A = 0
    THETA_B = 1
    THETA_C = 2
    THETA_D = 3
    DISP_KP1_X = 4
    DISP_KP1_P = 5
    DISP_KPN_X = 6
    DISP_KPN_P = 7


_generation_method_dict = OneToOneDict[FFCoeffMatrixGenerationMethods, PbFeedForwardCoefficientGenerationMethod](
    [
        (
            FFCoeffMatrixGenerationMethods.ZERO_FILLED,
            PbFeedForwardCoefficientGenerationMethod.FEED_FORWARD_COEFFICIENT_GENERATION_METHOD_ZERO_FILLED,
        ),
        (
            FFCoeffMatrixGenerationMethods.FROM_MACRONODE_ANGLE,
            PbFeedForwardCoefficientGenerationMethod.FEED_FORWARD_COEFFICIENT_GENERATION_METHOD_FROM_HOMODYNE_ANGLES,
        ),
    ],
)


class MachineryRepr:
    """Machinery representation of continuous variable quantum computing."""

    __n_local_macronodes: int
    __n_steps: int
    __readout_macronode_indices: set[int]
    __homodyne_angles: list[MacronodeAngle]
    __displacements_k_minus_1: list[tuple[MachineOpParam, MachineOpParam]]
    __displacements_k_minus_n: list[tuple[MachineOpParam, MachineOpParam]]
    ff_coeff_matrix_k_plus_1: _FFCoeffMatrixList
    ff_coeff_matrix_k_plus_n: _FFCoeffMatrixList

    def __init__(  # noqa:PLR0913
        self,
        n_local_macronodes: int,
        n_steps: int,
        *,
        readout_macronode_indices: set[int] | None = None,
        homodyne_angles: list[MacronodeAngle] | None = None,
        displacements_k_minus_1: list[tuple[MachineOpParam, MachineOpParam]] | None = None,
        displacements_k_minus_n: list[tuple[MachineOpParam, MachineOpParam]] | None = None,
    ) -> None:
        """Initialize a machinery representation.

        Args:
            n_local_macronodes (int): Number of local macronodes per step.
            n_steps (int): Number of steps.
            readout_macronode_indices (set[int] | None, optional): Set of macronode indices to get measurement results.
                Defaults to None.
            homodyne_angles (list[MacronodeAngle] | None, optional): Homodyne angles for each macronode.
                Defaults to None.
            displacements_k_minus_1 (list[tuple[MachineOpParam, MachineOpParam]] | None, optional):
                Displacements applied at k-1. Defaults to None.
            displacements_k_minus_n (list[tuple[MachineOpParam, MachineOpParam]] | None, optional):
                Displacements applied at k-n. Defaults to None.

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...     n_local_macronodes=2,
            ...     n_steps=3,
            ...     readout_macronode_indices={0, 1, 2},
            ...     displacements_k_minus_1=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
            ...     displacements_k_minus_n=[(0.7, 0.8), (0.9, 1.0), (1.1, 1.2)],
            ... )

        Raises:
            ValueError: If the length of a list is invalid.
        """
        self.__n_local_macronodes = n_local_macronodes
        self.__n_steps = n_steps

        if readout_macronode_indices is None:
            self.__readout_macronode_indices = set()
        else:
            self.__readout_macronode_indices = readout_macronode_indices

        if homodyne_angles is None:
            self.__homodyne_angles = [MacronodeAngle(0, 0, 0, 0)] * (n_local_macronodes * n_steps)
        else:
            if len(homodyne_angles) != n_local_macronodes * n_steps:
                msg = f"The number of homodyne angle vectors must be {self.n_local_macronodes * self.n_steps}."
                raise ValueError(msg)
            self.__homodyne_angles = homodyne_angles

        if displacements_k_minus_1 is None:
            self.__displacements_k_minus_1 = [(0, 0) for _ in range(self.n_local_macronodes * self.n_steps)]
        else:
            self.__displacements_k_minus_1 = displacements_k_minus_1
        if displacements_k_minus_n is None:
            self.__displacements_k_minus_n = [(0, 0) for _ in range(self.n_local_macronodes * self.n_steps)]
        else:
            self.__displacements_k_minus_n = displacements_k_minus_n

        self.ff_coeff_matrix_k_plus_1 = _FFCoeffMatrixList(
            calculate_ff_matrix_kp1,
            1,
            self.__homodyne_angles,
        )
        self.ff_coeff_matrix_k_plus_n = _FFCoeffMatrixList(
            calculate_ff_matrix_kpn,
            self.n_local_macronodes,
            self.__homodyne_angles,
        )

    @staticmethod
    def from_graph_repr(
        graph_repr: GraphRepr, *, empty_wire_policy: Literal["teleport", "measure"] = "teleport"
    ) -> MachineryRepr:
        """Convert graph representation to machinery representation.

        An *empty wire* is a macronode through which no mode passes (i.e., all
        I/O modes are ``BLANK_MODE``). By default such macronodes are realized as
        teleportation macronodes. When ``empty_wire_policy="measure"``, they are
        realized as **measurement macronodes without readout**, which is typically
        more hardware-friendly and avoids useless teleportation.

        Args:
            graph_repr (GraphRepr): Source graph representation.
            empty_wire_policy (Literal["teleport", "measure"], optional): {"teleport", "measure"}, default="teleport"
                How to materialize empty-wire macronodes:

                * "teleport": realize them as teleportation macronodes (default).
                * "measure": realize them as **measurement macronodes without readout**;
                    for those macronodes the homodyne angle is set to ``MacronodeAngle(0, 0, 0, 0)``
                    and the feedforward coefficient matrices (k+1, k+n) are marked as ``ZERO_FILLED``.

        Returns:
            MachineryRepr: Converted machinery representation.

        Notes:
            * **Logical semantics are unchanged**; choosing ``empty_wire_policy="measure"``
                only affects hardware-level realization and resource usage.
            * **No readout index** is added when ``empty_wire_policy="measure"``.

        Examples:
            >>> from mqc3.graph import GraphRepr, ops
            >>> from mqc3.graph.constant import BLANK_MODE
            >>> from mqc3.machinery import MachineryRepr
            >>> graph = GraphRepr(n_local_macronodes=2, n_steps=3)
            >>> op = ops.Initialization((0, 0), initialized_modes=[0, BLANK_MODE], theta=1)
            >>> graph.place_operation(op)
            >>> op = ops.PhaseRotation((0, 1), swap=True, phi=1)
            >>> graph.place_operation(op)
            >>> machinery = MachineryRepr.from_graph_repr(graph)
        """
        io_modes_dict = graph_repr.io_modes_dict()

        ret_repr = MachineryRepr(n_local_macronodes=graph_repr.n_local_macronodes, n_steps=graph_repr.n_steps)

        readout_macronode_indices: set[int] = set()
        for ind in range(graph_repr.n_total_macronodes):
            coord = graph_repr.get_coord(ind)
            op = graph_repr.get_operation(coord[0], coord[1])
            empty_wire = io_modes_dict[coord] == (BLANK_MODE, BLANK_MODE, BLANK_MODE, BLANK_MODE)

            # Set displacement.
            ret_repr.set_displacement_k_minus_1(
                ind,
                (
                    convert_gm_param(op.displacement_k_minus_1[0], graph_repr),
                    convert_gm_param(op.displacement_k_minus_1[1], graph_repr),
                ),
            )
            ret_repr.set_displacement_k_minus_n(
                ind,
                (
                    convert_gm_param(op.displacement_k_minus_n[0], graph_repr),
                    convert_gm_param(op.displacement_k_minus_n[1], graph_repr),
                ),
            )

            # Set homodyne_angle, ff_coeff_matrix and readout.
            if empty_wire_policy == "measure" and empty_wire:
                op_angle = MacronodeAngle(0, 0, 0, 0)
                ret_repr.set_homodyne_angle(ind, op_angle)
                ret_repr.ff_coeff_matrix_k_plus_1.generation_methods[ind] = FFCoeffMatrixGenerationMethods.ZERO_FILLED
                ret_repr.ff_coeff_matrix_k_plus_n.generation_methods[ind] = FFCoeffMatrixGenerationMethods.ZERO_FILLED
            else:
                op_angle = from_graph_operation(op, graph_repr)
                ret_repr.set_homodyne_angle(ind, op_angle)

                if isinstance(op, gops.Measurement):
                    ret_repr.ff_coeff_matrix_k_plus_1.generation_methods[ind] = (
                        FFCoeffMatrixGenerationMethods.ZERO_FILLED
                    )
                    ret_repr.ff_coeff_matrix_k_plus_n.generation_methods[ind] = (
                        FFCoeffMatrixGenerationMethods.ZERO_FILLED
                    )

                if op.readout:
                    readout_macronode_indices.add(graph_repr.get_index(*op.macronode))

        ret_repr.readout_macronode_indices = readout_macronode_indices

        return ret_repr

    def set_homodyne_angle(self, macronode_index: int, angle: MacronodeAngle) -> None:
        """Set homodyne angles in a macronode.

        Args:
            macronode_index (int): Index of the macronode.
            angle (MacronodeAngle): Homodyne angle.

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> from mqc3.machinery.macronode_angle import MacronodeAngle
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.set_homodyne_angle(0, MacronodeAngle(1, 1, 1, 1))
            >>> machinery_repr.get_homodyne_angle(0)
            MacronodeAngle(theta_a=1, theta_b=1, theta_c=1, theta_d=1)
        """
        self.__homodyne_angles[macronode_index] = angle
        self.ff_coeff_matrix_k_plus_1.homodyne_angles[macronode_index] = angle
        self.ff_coeff_matrix_k_plus_n.homodyne_angles[macronode_index] = angle

    def get_homodyne_angle(self, macronode_index: int) -> MacronodeAngle:
        """Get homodyne angle of a macronode.

        Args:
            macronode_index (int): Index of the macronode

        Returns:
            MacronodeAngle: Homodyne angle.

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> from mqc3.machinery.macronode_angle import MacronodeAngle
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.set_homodyne_angle(0, MacronodeAngle(1, 1, 1, 1))
            >>> machinery_repr.get_homodyne_angle(0)
            MacronodeAngle(theta_a=1, theta_b=1, theta_c=1, theta_d=1)
        """
        return self.__homodyne_angles[macronode_index]

    def get_coord(self, i: int) -> tuple[int, int]:
        """Get the coordinate of an index.

        Args:
            i (int): 0 <= `i` < `n_total_macronodes`.

        Returns:
            tuple[int, int]: The coordinate of `i`-th macronode.

        Raises:
            IndexError: The given `i` is out of range.

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...     n_local_macronodes=2,
            ...     n_steps=3,
            ...     readout_macronode_indices={0, 1, 2},
            ...     displacements_k_minus_1=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
            ...     displacements_k_minus_n=[(0.7, 0.8), (0.9, 1.0), (1.1, 1.2)],
            ... )
            >>> machinery_repr.get_coord(i=5)
            (1, 2)
        """
        if i < 0 or i >= self.n_total_macronodes:
            msg = f"`i` must be in the range [0, {self.n_total_macronodes})."
            raise IndexError(msg)

        w, h = divmod(i, self.n_local_macronodes)
        return (h, w)

    @property
    def n_local_macronodes(self) -> int:
        """Get number of local macronodes.

        Returns:
            int: The number of local macronodes.

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.n_local_macronodes
            2
        """
        return self.__n_local_macronodes

    @property
    def n_steps(self) -> int:
        """Get number of steps.

        Returns:
            int: The number of steps.

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.n_steps
            3
        """
        return self.__n_steps

    @property
    def n_total_macronodes(self) -> int:
        """Get total number of macronodes.

        The total number of macronodes can be calculated as: `self.n_local_macronodes * self.n_steps`.

        Returns:
            int: Total number of macronodes.

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.n_total_macronodes
            6
        """
        return self.n_local_macronodes * self.n_steps

    @property
    def readout_macronode_indices(self) -> list[int]:
        """Get the indices of readout macronodes.

        Returns:
            list[int]: The indices of readout macronodes, sorted in ascending order.

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=2,
            ...    n_steps=3,
            ...    readout_macronode_indices={0, 1, 2},
            ... )
            >>> machinery_repr.readout_macronode_indices
            [0, 1, 2]
        """
        return sorted(self.__readout_macronode_indices)

    @readout_macronode_indices.setter
    def readout_macronode_indices(self, val: set[int]) -> None:
        self.__readout_macronode_indices = val

    @property
    def displacements_k_minus_1(self) -> list[tuple[MachineOpParam, MachineOpParam]]:
        """Get the displacement vector for k-1.

        Returns:
            list[tuple[MachineOpParam, MachineOpParam]]: Displacement vector

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=1,
            ...    n_steps=2,
            ...    readout_macronode_indices={1},
            ... )
            >>> machinery_repr.displacements_k_minus_1
            [(0, 0), (0, 0)]
        """
        return self.__displacements_k_minus_1

    @displacements_k_minus_1.setter
    def displacements_k_minus_1(self, val: list[tuple[MachineOpParam, MachineOpParam]]) -> None:
        if len(val) != self.__n_local_macronodes * self.__n_steps:
            message = f"The number of displacement vectors must be {self.__n_local_macronodes * self.__n_steps}.\n"
            message += f"{len(val)} vectors are given."
            raise ValueError(message)
        for vector in val:
            if len(vector) != 2:  # noqa: PLR2004
                message = "Each displacement vector must have two elements."
                raise ValueError(message)
        self.__displacements_k_minus_1 = val

    def set_displacement_k_minus_1(self, macronode_index: int, val: tuple[MachineOpParam, MachineOpParam]) -> None:
        """Set the displacement applied to the mode teleporting from macronode k-1 to macronode k.

        Args:
            macronode_index (int): Index of the macronode k.
            val (tuple[MachineOpParam, MachineOpParam]): Displacement parameters to be set.
        """
        self.displacements_k_minus_1[macronode_index] = val

    def get_displacement_k_minus_1(self, macronode_index: int) -> tuple[MachineOpParam, MachineOpParam]:
        """Get the displacement applied to the mode teleporting from macronode k-1 to macronode k.

        Args:
            macronode_index (int): Index of the macronode k.

        Returns:
            tuple[MachineOpParam, MachineOpParam]: Displacement parameters for the macronode.
        """
        return self.displacements_k_minus_1[macronode_index]

    @property
    def displacements_k_minus_n(self) -> list[tuple[MachineOpParam, MachineOpParam]]:
        """Get the displacement vector for k-N.

        Returns:
            list[tuple[MachineOpParam, MachineOpParam]]: Displacement vector.

        Examples:
            >>> from mqc3.machinery import MachineryRepr
            >>> machinery_repr = MachineryRepr(
            ...    n_local_macronodes=1,
            ...    n_steps=2,
            ...    readout_macronode_indices={1},
            ... )
            >>> machinery_repr.displacements_k_minus_n
            [(0, 0), (0, 0)]
        """
        if self.__displacements_k_minus_n is None:
            self.__displacements_k_minus_n = [(0, 0) for _ in range(self.n_local_macronodes * self.n_steps)]

        return self.__displacements_k_minus_n

    @displacements_k_minus_n.setter
    def displacements_k_minus_n(self, val: list[tuple[MachineOpParam, MachineOpParam]]) -> None:
        if len(val) != self.__n_local_macronodes * self.__n_steps:
            message = f"The number of displacement vectors must be {self.__n_local_macronodes * self.__n_steps}.\n"
            message += f"{len(val)} vectors are given."
            raise ValueError(message)
        for vector in val:
            if len(vector) != 2:  # noqa: PLR2004
                message = "Each displacement vector must have two elements"
                raise ValueError(message)
        self.__displacements_k_minus_n = val

    def set_displacement_k_minus_n(self, macronode_index: int, val: tuple[MachineOpParam, MachineOpParam]) -> None:
        """Set the displacement applied to the mode teleporting from macronode k-N to macronode k.

        Args:
            macronode_index (int): Index of the macronode k.
            val (tuple[MachineOpParam, MachineOpParam]): Displacement parameters to be set.
        """
        self.displacements_k_minus_n[macronode_index] = val

    def get_displacement_k_minus_n(self, macronode_index: int) -> tuple[MachineOpParam, MachineOpParam]:
        """Get the displacement applied to the mode teleporting from macronode k-N to macronode k.

        Args:
            macronode_index (int): Index of the macronode k.

        Returns:
            tuple[MachineOpParam, MachineOpParam]: Displacement parameters for the macronode.
        """
        return self.displacements_k_minus_n[macronode_index]

    def proto(self) -> PbMachineryRepr:  # noqa: D102
        def _convert_to_pb_displacement(
            vector: tuple[MachineOpParam, MachineOpParam],
        ) -> PbMachineryRepr.DisplacementComplex:
            if not isinstance(vector, tuple):
                raise TypeError
            if len(vector) != 2:  # noqa: PLR2004
                msg = "The input tuple must have two elements."
                raise ValueError(msg)
            return PbMachineryRepr.DisplacementComplex(
                x=vector[0] if not isinstance(vector[0], FeedForward) else 0.0,
                p=vector[1] if not isinstance(vector[1], FeedForward) else 0.0,
            )

        homodyne_angles: list[PbMachineryRepr.MacronodeAngle] = []
        funcs: list[PbPythonFunction] = []
        nlffs: list[PbMachineryFF] = []
        for i in range(self.n_total_macronodes):
            angle = self.get_homodyne_angle(i)

            km1 = self.displacements_k_minus_1[i]
            kmn = self.displacements_k_minus_n[i]

            homodyne_angles.append(angle.proto())

            pos_list = [angle.theta_a, angle.theta_b, angle.theta_c, angle.theta_d, km1[0], km1[1], kmn[0], kmn[1]]
            for ind, ff in zip(_FFPosition, pos_list, strict=False):
                if not isinstance(ff, FeedForward):
                    continue

                from_ind, abcd = ff.variable.get_from_operation()

                func = ff.func.proto()
                index = -1
                for j, tmp in enumerate(funcs):
                    if str(func) == str(tmp):
                        index = j
                        break
                if index == -1:
                    funcs.append(func)
                    index = len(funcs) - 1

                nlffs.append(
                    PbMachineryFF(
                        function=index,
                        from_macronode=from_ind,
                        from_abcd=abcd,
                        to_macronode=i,
                        to_parameter=ind,
                    ),
                )

        kp1_methods = [_generation_method_dict.get_v(k) for k in self.ff_coeff_matrix_k_plus_1.generation_methods]
        kpn_methods = [_generation_method_dict.get_v(k) for k in self.ff_coeff_matrix_k_plus_n.generation_methods]

        return PbMachineryRepr(
            n_local_macronodes=self.n_local_macronodes,
            n_steps=self.n_steps,
            homodyne_angles=homodyne_angles,
            generating_method_for_ff_coeff_k_plus_1=kp1_methods,
            generating_method_for_ff_coeff_k_plus_n=kpn_methods,
            displacements_k_minus_1=[_convert_to_pb_displacement(d) for d in self.displacements_k_minus_1],
            displacements_k_minus_n=[_convert_to_pb_displacement(d) for d in self.displacements_k_minus_n],
            readout_macronodes_indices=self.readout_macronode_indices,
            functions=funcs,
            nlffs=nlffs,
        )

    @staticmethod
    def construct_from_proto(proto: PbMachineryRepr) -> MachineryRepr:  # noqa: D102
        empty_macronode_angle = MacronodeAngle(0, 0, 0, 0).proto()

        machinery_repr = MachineryRepr(
            n_local_macronodes=proto.n_local_macronodes,
            n_steps=proto.n_steps,
            readout_macronode_indices=set(proto.readout_macronodes_indices),
        )

        # homodyne_angles
        for i, proto_angle in enumerate(proto.homodyne_angles):
            if proto_angle == empty_macronode_angle:
                machinery_repr.set_homodyne_angle(macronode_index=i, angle=MacronodeAngle(0, 0, 0, 0))
                continue

            machinery_repr.set_homodyne_angle(
                macronode_index=i,
                angle=MacronodeAngle.construct_from_proto(proto_angle),
            )

        # ff coefficient matrices
        for i, method in enumerate(proto.generating_method_for_ff_coeff_k_plus_1):
            machinery_repr.ff_coeff_matrix_k_plus_1.generation_methods[i] = _generation_method_dict.get_k(method)

        for i, method in enumerate(proto.generating_method_for_ff_coeff_k_plus_n):
            machinery_repr.ff_coeff_matrix_k_plus_n.generation_methods[i] = _generation_method_dict.get_k(method)

        # displacements matrices
        machinery_repr.displacements_k_minus_1 = [
            (proto_displacement.x, proto_displacement.p) for proto_displacement in proto.displacements_k_minus_1
        ]
        machinery_repr.displacements_k_minus_n = [
            (proto_displacement.x, proto_displacement.p) for proto_displacement in proto.displacements_k_minus_n
        ]

        for nlff in proto.nlffs:
            ff_func = FeedForwardFunction.construct_from_proto(proto.functions[nlff.function])
            var = MeasuredVariable(nlff.from_macronode, nlff.from_abcd)

            ff = ff_func(var)
            angle = machinery_repr.get_homodyne_angle(nlff.to_macronode)
            if angle is None:
                msg = f"Macronode {nlff.to_macronode} is not defined."
                raise ValueError(msg)
            machinery_repr.set_homodyne_angle(
                nlff.to_macronode,
                MacronodeAngle(
                    theta_a=ff if nlff.to_parameter == _FFPosition.THETA_A else angle.theta_a,
                    theta_b=ff if nlff.to_parameter == _FFPosition.THETA_B else angle.theta_b,
                    theta_c=ff if nlff.to_parameter == _FFPosition.THETA_C else angle.theta_c,
                    theta_d=ff if nlff.to_parameter == _FFPosition.THETA_D else angle.theta_d,
                ),
            )
            if nlff.to_parameter in {_FFPosition.DISP_KP1_X, _FFPosition.DISP_KP1_P}:
                km1 = machinery_repr.displacements_k_minus_1
                km1[nlff.to_macronode] = (
                    ff if nlff.to_parameter == _FFPosition.DISP_KP1_X else km1[nlff.to_macronode][0],
                    ff if nlff.to_parameter == _FFPosition.DISP_KP1_P else km1[nlff.to_macronode][1],
                )

                machinery_repr.displacements_k_minus_1 = km1
            if nlff.to_parameter in {_FFPosition.DISP_KPN_X, _FFPosition.DISP_KPN_P}:
                kmn = machinery_repr.displacements_k_minus_n
                kmn[nlff.to_macronode] = (
                    ff if nlff.to_parameter == _FFPosition.DISP_KPN_X else kmn[nlff.to_macronode][0],
                    ff if nlff.to_parameter == _FFPosition.DISP_KPN_P else kmn[nlff.to_macronode][1],
                )

                machinery_repr.displacements_k_minus_n = kmn

        return machinery_repr

    def save(self, path: str | Path, proto_format: ProtoFormat = "text") -> None:  # noqa: D102
        save(self.proto(), path, proto_format)

    @staticmethod
    def load(path: str | Path, proto_format: ProtoFormat = "text") -> MachineryRepr:  # noqa: D102
        return MachineryRepr.construct_from_proto(load(PbMachineryRepr, path, proto_format))
