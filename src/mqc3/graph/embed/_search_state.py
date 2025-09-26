from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import mqc3.graph.ops as gops
from mqc3.feedforward import FeedForward
from mqc3.graph import GraphRepr
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.embed.embed import GraphEmbedSettings
from mqc3.graph.ops import ModeMeasuredVariable

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from mqc3.graph import Operation as GraphOp
    from mqc3.graph.embed.dep_dag import DependencyDAG


def _apply_ignored_none(
    a: int | None,
    b: int | None,
    func: Callable[[int, int], int],
) -> int:
    if a is not None and b is not None:
        return func(a, b)
    if a is None and b is not None:
        return b
    if a is not None and b is None:
        return a

    msg = "Both `a` and `b` are `None`."
    raise RuntimeError(msg)


def get_min_ignored_none(a: int | None, b: int | None) -> int:
    return _apply_ignored_none(a, b, min)


def get_max_ignored_none(a: int | None, b: int | None) -> int:
    return _apply_ignored_none(a, b, max)


class SearchState:
    __slots__ = [
        "_left",
        "_mode_pos",
        "_op_pos_dict",
        "_swap_pos_set",
        "_up",
        "dep_dag",
        "feedforward_distance",
        "index",
        "mode_to_measurement",
        "n_local_macronodes",
    ]

    def __init__(self, dep_dag: DependencyDAG | None, settings: GraphEmbedSettings) -> None:
        """Search state for the embedding.

        Args:
            dep_dag (DependencyDAG): The dependency DAG to embed. If None, should be set later.
            settings (GraphEmbedConfig): The settings of embedding.

        Attributes:
            _mode_pos (dict[int, int]): Current position of a mode.
                                         - If `_left[h] == mode`: h
                                         - If `_up == mode` : -1
                                         - Otherwise : None
        """
        if dep_dag is None:
            return

        # Constant
        self.n_local_macronodes = settings.n_local_macronodes
        self.feedforward_distance = settings.feedforward_distance
        self.dep_dag = dep_dag
        self.mode_to_measurement: dict[int, int] = {}
        for ind, node in self.dep_dag.dag.nodes.items():
            if isinstance(node["op"], gops.Measurement):
                self.mode_to_measurement[node["modes"][0]] = ind

        # Position of operations
        self._swap_pos_set: set[tuple[int, int]] = set()
        self._op_pos_dict: dict[int, tuple[int, int]] = {}

        # Macronode index
        self.index = 0

        # Mode
        self._up = BLANK_MODE
        self._left = [BLANK_MODE] * self.n_local_macronodes
        self._mode_pos: dict[int, int] = {}

    def get_left_mode(self, index: int) -> int:
        """Get the left mode of the `index`-th macronode.

        Args:
            index (int): Index of the macronode.

        Returns:
            int: The left mode.
        """
        return self._left[index % self.n_local_macronodes]

    def get_coord(self, index: int) -> tuple[int, int]:
        """Get the coordinate of the `index`-th macronode.

        Args:
            index (int): Index of the macronode.

        Returns:
            tuple[int, int]: Coordinate.
        """
        w, h = divmod(index, self.n_local_macronodes)
        return h, w

    def get_index(self, coord: tuple[int, int]) -> int:
        """Get the index of coordinate.

        Args:
            coord (int): coordinate of the macronode

        Returns:
            int: index
        """
        return coord[1] * self.n_local_macronodes + coord[0]

    def search_mode(self, mode: int) -> int:
        """Get the index of the macronode with the specified mode after the current macronode.

        Args:
            mode (int): Mode.

        Raises:
            RuntimeError: There is no macronode with the specified mode.

        Returns:
            int: Macronode index.
        """
        if self._up == mode:
            return self.index

        from_left = self.find_mode_from_left(mode)
        if from_left is not None:
            return from_left

        msg = f"There is no mode: {mode}."
        raise RuntimeError(msg)

    def find_mode_from_left(self, mode: int) -> int | None:
        """Get the index of the macronode, that the specified mode come from left, after the current macronode.

        Args:
            mode (int): Mode.

        Returns:
            int | None: Macronode index or None if not found.
        """
        if mode != BLANK_MODE:
            if mode not in self._mode_pos or self._mode_pos[mode] == -1:
                return None
            h = self._mode_pos[mode]
            return self.index + (h - self.index) % self.n_local_macronodes
        for i in range(self.index, self.index + self.n_local_macronodes):
            if self.get_left_mode(i) == mode:
                return i
        return None

    def blank_mode_count(self) -> int:
        return self.n_local_macronodes + 1 - len(self._mode_pos)

    def calc_min_index_to_place_op(self, op_ind: int) -> int | None:
        """Calculate the minimum finish time of `op`.

        Returns:
            int | None : Minimum finish time of `op`. If the op cannot be placed currently, returns None.
        """
        if isinstance(self.dep_dag.dag.nodes[op_ind]["op"], gops.Initialization):
            found_first_blank = False
            if self._up == BLANK_MODE:
                found_first_blank = True
            for i in range(self.index, self.index + self.n_local_macronodes):
                if self._left[i % self.n_local_macronodes] == BLANK_MODE:
                    if found_first_blank:
                        return i
                    found_first_blank = True
            return None  # Two blank modes are required to place initialization.
        modes = self.dep_dag.dag.nodes[op_ind]["modes"]
        if len(modes) == 1:
            mode = modes[0]
            blank_index = self.search_mode(BLANK_MODE)
            mode_index = self.search_mode(mode)
            return max(blank_index, mode_index)

        target_mode1, target_mode2 = modes
        target_mode1_index = self.search_mode(target_mode1)
        target_mode2_index = self.search_mode(target_mode2)
        target_mode1_h = self.get_coord(target_mode1_index)[0]
        target_mode2_h = self.get_coord(target_mode2_index)[0]
        upper_mode_index, lower_mode_index = (
            (target_mode1_index, target_mode2_index)
            if target_mode1_h < target_mode2_h
            else (target_mode2_index, target_mode1_index)
        )
        return lower_mode_index + (self.n_local_macronodes if lower_mode_index < upper_mode_index else 0)

    def advance_index(self) -> None:
        """Increment index and call functions."""
        self.index += 1

    def remove_current_modes(self) -> None:
        """Remove modes on current macronode from SearchState."""
        self._mode_pos.pop(self._up, None)
        self._up = BLANK_MODE

        h, _ = self.get_coord(self.index)
        self._mode_pos.pop(self._left[h], None)
        self._left[h] = BLANK_MODE

    def insert_through(self, reps: int = 1, *, without_leap: bool = False) -> None:
        """Insert `through` operations.

        If "without_leap" is True, it also inserts minimum required `swap` operations to avoid leap of modes.

        Args:
            reps (int): The number of iterations.
            without_leap (bool): Avoid leap of modes.
        """
        for _ in range(reps):
            if without_leap and self._up != BLANK_MODE and self.get_left_mode(self.index) == BLANK_MODE:
                self.insert_swap()
            else:
                self.advance_index()

    def insert_swap(self) -> None:
        """Insert the `swap` operation.

        Raises:
            RuntimeError: The swap operation is already placed
        """
        coord = self.get_coord(self.index)
        if coord in self._swap_pos_set:
            msg = "The swap operation is already placed."
            raise RuntimeError(msg)
        self._swap_pos_set.add(coord)

        h = self.get_coord(self.index)[0]
        self._up, self._left[h] = self._left[h], self._up
        if self._up != BLANK_MODE:
            self._mode_pos[self._up] = -1
        if self._left[h] != BLANK_MODE:
            self._mode_pos[self._left[h]] = h

        self.advance_index()

    def insert_initialization(self, op_ind: int, *, swap_in_op: bool) -> None:
        """Insert initialization operation.

        Args:
            op_ind (int): Inserting operation index in the dependency graph.
            swap_in_op (bool): Whether the swap operation is included in the operation.
        """
        op = self.dep_dag.dag.nodes[op_ind]["op"]
        h = self.get_coord(self.index)[0]
        if op.initialized_modes[0] != BLANK_MODE:
            new_mode = op.initialized_modes[0]
            self._up = new_mode
            self._mode_pos[new_mode] = -1
        if op.initialized_modes[1] != BLANK_MODE:
            new_mode = op.initialized_modes[1]
            self._left[h] = new_mode
            self._mode_pos[new_mode] = h
        self._op_pos_dict[op_ind] = self.get_coord(self.index)
        if swap_in_op:
            self.insert_swap()
        else:
            self.insert_through()

    def insert_single_mode_operation(self, op_ind: int, *, swap_in_op: bool) -> None:
        """Insert the single-mode operation.

        Args:
            op_ind (int): Inserting operation index in the dependency graph.
            swap_in_op (bool): Whether the swap operation is included in the operation.

        Raises:
            RuntimeError: Modes are not prepared correctly
        """
        left = self.get_left_mode(self.index)
        up = self._up

        modes = self.dep_dag.dag.nodes[op_ind]["modes"]
        if BLANK_MODE not in {left, up}:
            msg = "One of the input modes of the current macronode must be blank mode."
            raise RuntimeError(msg)
        if modes[0] not in {left, up}:
            msg = "The input mode of the operation must match the non-blank mode of the macronode."
            raise RuntimeError(msg)

        op = self.dep_dag.dag.nodes[op_ind]["op"]
        self._op_pos_dict[op_ind] = self.get_coord(self.index)

        # remove measured modes
        if isinstance(op, gops.Measurement):
            self.remove_current_modes()
        # Up mode should be kept blank as possible
        if swap_in_op:
            self.insert_swap()
        else:
            self.insert_through()

    def insert_two_mode_operation(self, op_ind: int, *, swap_in_op: bool) -> None:
        """Insert the two-mode operation.

        Args:
            op_ind (int): Inserting operation index in the dependency graph.
            swap_in_op (bool): Whether the swap operation is included in the operation.

        Raises:
            RuntimeError: Input modes of the operation and the modes of the macronode do not match.
        """
        modes = self.dep_dag.dag.nodes[op_ind]["modes"]

        left = self.get_left_mode(self.index)
        up = self._up

        if modes[0] not in {left, up} or modes[1] not in {left, up}:
            msg = "The first input mode of the operation must be the same as the left mode of the macronode."
            raise RuntimeError(msg)

        self._op_pos_dict[op_ind] = self.get_coord(self.index)
        if swap_in_op:
            self.insert_swap()
        else:
            self.insert_through()

    def is_all_dependency_resolved(self, op_ind: int) -> bool:
        return all(n in self._op_pos_dict for n in self.dep_dag.dag.predecessors(op_ind))

    def is_already_placed(self, op_ind: int) -> bool:
        return op_ind in self._op_pos_dict

    def calc_placeable_range(self, op_ind: int) -> tuple[int, int] | None:
        """Calculate the placeable range of an operation.

        Returns:
            tuple[int, int] | None: the lower and upper bound of the placeable range, inclusive.
                if the operation cannot be placed currently, returns None.

        Raises:
            TypeError: The type of parameters of operation does not match
            ValueError: The mode of ModeMeasuredVariable does not match
        """
        if not self.is_all_dependency_resolved(op_ind):
            return None  # Some of dependencies are not resolved
        inf = 10**9
        min_index = -1
        max_index = inf
        op = self.dep_dag.dag.nodes[op_ind]["op"]
        for p in op.parameters:
            if not isinstance(p, FeedForward):
                continue
            if not isinstance(p.variable, ModeMeasuredVariable):
                msg = "The type of parameters of operation does not match."
                raise TypeError(msg)
            if p.variable.mode not in self.mode_to_measurement:
                msg = "The mode of ModeMeasuredVariable does not match."
                raise ValueError(msg)
            measurement_ind = self.mode_to_measurement[p.variable.mode]

            # Check if the measurement for variable is already placed
            from_position = self._op_pos_dict.get(measurement_ind)
            if from_position is None:
                return None  # Some of operations required for feedforwarding are not placed
            if isinstance(op, gops.Initialization) and self.blank_mode_count() <= 1:
                return None  # There must be two or more blank modes to place initialization
            from_index = self.get_index(from_position)
            min_index = max(min_index, from_index + self.feedforward_distance[0])
            max_index = min(max_index, from_index + self.feedforward_distance[1])
        return (min_index, max_index)

    def place_operation(self, op_ind: int, *, swap_in_op: bool) -> None:
        if op_ind in self._op_pos_dict:
            msg = "This operation has been already placed"
            raise RuntimeError(msg)
        placeable_range = self.calc_placeable_range(op_ind)
        if placeable_range is None:
            msg = "This operation is not placeable currently."
            raise RuntimeError(msg)
        min_index, max_index = placeable_range
        while True:
            placement_index = self.calc_min_index_to_place_op(op_ind)
            if placement_index is None:
                msg = "This operation is not placeable currently."
                raise RuntimeError(msg)
            if placement_index >= min_index:
                break
            self.insert_through()
        if placement_index > max_index:
            msg = "Failed in insertion of operation."
            raise RuntimeError(msg)
        modes = self.dep_dag.dag.nodes[op_ind]["modes"]
        op = self.dep_dag.dag.nodes[op_ind]["op"]
        if isinstance(op, gops.Initialization):
            self.prepare_initialization()
            self.insert_initialization(op_ind, swap_in_op=swap_in_op)
        elif len(modes) == 1:
            self.prepare_single_mode_operation(modes[0])
            self.insert_single_mode_operation(op_ind, swap_in_op=swap_in_op)
        else:
            self.prepare_two_mode_operation(modes[0], modes[1])
            self.insert_two_mode_operation(op_ind, swap_in_op=swap_in_op)

    def generate_next_states(self) -> Generator[SearchState, None, None]:
        for op_ind in self.dep_dag.dag.nodes:
            if self.is_already_placed(op_ind) or not self.is_all_dependency_resolved(op_ind):
                continue
            placeable_range = self.calc_placeable_range(op_ind)
            if placeable_range is None:
                continue
            _min_index, max_index = placeable_range
            if self.index > max_index:  # conversion already failed
                return

        for op_ind, swap_in_op in product(self.dep_dag.dag.nodes, [False, True]):
            if self.is_already_placed(op_ind) or not self.is_all_dependency_resolved(op_ind):
                continue
            next_state = self.copy()
            next_state.place_operation(op_ind, swap_in_op=swap_in_op)
            yield next_state

    def output_graph(self) -> GraphRepr:  # noqa: C901
        """Output current state as `GraphRepr`.

        Returns:
            GraphRepr: Current state.
        """
        n_steps = 0
        for _, w in self._op_pos_dict.values():
            n_steps = max(n_steps, w + 1)
        for _, w in self._swap_pos_set:
            n_steps = max(n_steps, w + 1)

        graph = GraphRepr(self.n_local_macronodes, n_steps)

        for ind, coord in self._op_pos_dict.items():
            op: GraphOp = self.dep_dag.dag.nodes[ind]["op"]
            op.macronode = coord
            graph.place_operation(op)

        for coord in self._swap_pos_set:
            op = graph.get_operation(*coord)
            if isinstance(op, gops.Initialization):  # Initialization does not have swap attribute
                op.initialized_modes[0], op.initialized_modes[1] = op.initialized_modes[1], op.initialized_modes[0]
            else:
                op.swap = True

        io_modes_dict = None  # to improve performance, dict is created only when necessary

        for ind, coord in self._op_pos_dict.items():
            op = graph.get_operation(*coord)
            for mode, disps in self.dep_dag.dag.nodes[ind]["displacements"]:
                if io_modes_dict is None:
                    io_modes_dict = graph.io_modes_dict()
                left, up, _, _ = io_modes_dict[coord]
                if mode == left:
                    op.displacement_k_minus_n = disps
                elif mode == up:
                    op.displacement_k_minus_1 = disps

        return graph

    def evaluate(self) -> int:
        """Return the evaluate value of current state (larger means better)."""
        return len(self._op_pos_dict)

    def __lt__(self, other: SearchState) -> int:
        return self.evaluate() < other.evaluate()

    def is_all_done(self) -> bool:
        """Return true if all operations are done."""
        return all(n in self._op_pos_dict for n in self.dep_dag.dag.nodes)

    def prepare_initialization(self) -> None:
        """Use through or swap to create a macronode with input modes as (blank, blank).

        Raises:
            RuntimeError: Two blank nodes are required to place initialization.
        """
        first_blank_index = self.search_mode(BLANK_MODE)
        self.insert_through(first_blank_index - self.index)
        if self._up == BLANK_MODE and self.get_left_mode(self.index) == BLANK_MODE:
            return
        if self._up != BLANK_MODE:
            self.insert_swap()
        second_blank_index = self.find_mode_from_left(BLANK_MODE)
        if second_blank_index is None:
            msg = "Two blank nodes are required to place initialization."
            raise RuntimeError(msg)
        self.insert_through(second_blank_index - self.index)

    def prepare_single_mode_operation(self, target_mode: int) -> None:
        """Use through or swap to create a macronode with input modes as (blank, target).

        Args:
            target_mode (int): Mode used for operation.

        Raises:
            RuntimeError: Mode is not found.
        """
        blank_index = self.search_mode(BLANK_MODE)
        mode_index = self.find_mode_from_left(target_mode)

        if self._up != target_mode and self.get_left_mode(self.index) != target_mode and mode_index is None:
            msg = f"`target_mode` {target_mode} is not found."
            raise RuntimeError(msg)

        if self._up not in {target_mode, BLANK_MODE}:
            start_index = self.index
            next_index = get_min_ignored_none(mode_index, blank_index)
            self.insert_through(next_index - start_index, without_leap=True)
            self.insert_swap()

        if self._up not in {target_mode, BLANK_MODE}:
            msg = f"`target_mode` {target_mode} is not found."
            raise RuntimeError(msg)

        start_index = self.index
        next_index = get_max_ignored_none(mode_index, blank_index)
        self.insert_through(next_index - start_index)

    def prepare_two_mode_operation(self, target_mode1: int, target_mode2: int) -> None:
        """Place `through` or `swap` operations to make space for the target two-mode operation.

        Args:
            target_mode1 (int): The first input mode of the operation.
            target_mode2 (int): The second input mode of the operation.

        Raises:
            RuntimeError: The target two-mode operation cannot be placed.
        """
        # Is already prepared
        if (self._up == target_mode1 and self.get_left_mode(self.index) == target_mode2) or (
            self._up == target_mode2 and self.get_left_mode(self.index) == target_mode1
        ):
            return

        if self._up in {target_mode1, target_mode2}:
            other_mode = target_mode2 if self._up == target_mode1 else target_mode1
            other_mode_index = self.find_mode_from_left(other_mode)
            if other_mode_index is None:
                msg = f"{other_mode} is not found."
                raise RuntimeError(msg)
            if self.get_coord(other_mode_index)[0] >= self.get_coord(self.index)[0]:
                self.insert_through(other_mode_index - self.index)
                return
            self.insert_swap()

        target_mode1_index = self.find_mode_from_left(target_mode1)
        if target_mode1_index is None:
            msg = f"{target_mode1} is not found."
            raise RuntimeError(msg)

        target_mode2_index = self.find_mode_from_left(target_mode2)
        if target_mode2_index is None:
            msg = f"{target_mode2} is not found."
            raise RuntimeError(msg)

        target_mode1_h = self.get_coord(target_mode1_index)[0]
        target_mode2_h = self.get_coord(target_mode2_index)[0]
        upper_mode_index, lower_mode_index = (
            (target_mode1_index, target_mode2_index)
            if target_mode1_h < target_mode2_h
            else (target_mode2_index, target_mode1_index)
        )
        prepare_index = lower_mode_index + (self.n_local_macronodes if lower_mode_index < upper_mode_index else 0)

        self.insert_through(upper_mode_index - self.index, without_leap=True)
        self.insert_swap()
        self.insert_through(prepare_index - self.index)

    def copy(self) -> SearchState:
        """Copy the search state.

        Returns:
            SearchState: Copied search state.
        """
        copied = SearchState(None, GraphEmbedSettings(self.n_local_macronodes, self.feedforward_distance))

        copied._left = self._left.copy()
        copied._mode_pos = self._mode_pos.copy()
        copied._swap_pos_set = self._swap_pos_set.copy()
        copied._op_pos_dict = self._op_pos_dict.copy()
        copied._up = self._up
        copied.index = self.index
        copied.n_local_macronodes = self.n_local_macronodes
        copied.feedforward_distance = self.feedforward_distance
        copied.dep_dag = self.dep_dag
        copied.mode_to_measurement = self.mode_to_measurement

        return copied
