"""Dependency DAG constructor.

Dependency DAG is a Directed Acyclic Graph that represents the dependency of operations.
This module provides a class to construct the dependency graph from a quantum circuit.
"""

import math
from collections import defaultdict
from math import pi

import networkx as nx

import mqc3.circuit.ops.intrinsic as cops
import mqc3.graph.ops as gops
from mqc3.circuit.program import CircOpParam, CircuitRepr
from mqc3.circuit.state import HardwareConstrainedSqueezedState
from mqc3.feedforward import FeedForward, ff_to_add_constant
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.embed._utility import _convert_cg_param, convert_op
from mqc3.graph.ops import GraphOpParam, Measurement, ModeMeasuredVariable
from mqc3.graph.program import GraphRepr
from mqc3.graph.program import Operation as GraphOp
from mqc3.pb.mqc3_cloud.program.v1.graph_pb2 import GraphOperation as PbOperation

DEFAULT_COORDINATE = (-1, -1)


def _add_c_disp_to_g_disp(disp1: CircOpParam, disp2: GraphOpParam) -> GraphOpParam:
    """Construct a new displacement for graph by adding two displacements.

    Args:
        disp1 (CircOpParam): Displacement 1.
        disp2 (GraphOpParam): Displacement 2.

    Returns:
        GraphOpParam: New displacement.

    Raises:
        TypeError: Applying multiple displacement with feedforward is not allowed.
    """
    if isinstance(disp1, FeedForward):
        if isinstance(disp2, FeedForward):
            msg = "Applying multiple displacements with feedforward is not allowed."
            raise TypeError(msg)

        x1 = _convert_cg_param(disp1)
        x2 = disp2
    else:
        x1 = disp2
        x2 = disp1

    ff_func = ff_to_add_constant(x2)

    return ff_func(x1)


def _add_g_disps(disp1: GraphOpParam, disp2: GraphOpParam) -> GraphOpParam:
    """Construct a new displacement for graph by adding two displacements.

    Args:
        disp1 (GraphOpParam): Displacement 1.
        disp2 (GraphOpParam): Displacement 2.

    Returns:
        GraphOpParam: New displacement.

    Raises:
        TypeError: Applying multiple displacement with feedforward is not allowed.
    """
    if isinstance(disp1, FeedForward):
        if isinstance(disp2, FeedForward):
            msg = "Applying multiple displacements with feedforward is not allowed."
            raise TypeError(msg)
        x1, x2 = disp1, disp2
    else:
        x1, x2 = disp2, disp1

    return ff_to_add_constant(x2)(x1)


class DependencyDAG:
    """Directed Acyclic Graph which represents dependency relationship between operations."""

    def __init__(self, circuit_or_graph: CircuitRepr | GraphRepr) -> None:
        """Initializes the DependencyDAG by constructing its internal dependency graph.

        Args:
            circuit_or_graph (CircuitRepr | GraphRepr):
                The input quantum circuit representation or graph representation
                from which to build the DependencyDAG.
        """
        self.dag: nx.DiGraph

        builder = _DependencyBuilder()
        if isinstance(circuit_or_graph, CircuitRepr):
            self.dag = builder.from_circuit(circuit_or_graph)
        else:
            self.dag = builder.from_graph(circuit_or_graph)
        self.n_modes = len(builder.last_op_of_modes)


class _DependencyBuilder:
    next_node_id: int
    dep_graph: nx.DiGraph
    last_disp_of_modes: dict[int, tuple[GraphOpParam, GraphOpParam] | None]
    last_op_of_modes: dict[int, int]
    measurement_to_mode: dict[int, int]

    def __init__(self) -> None:
        pass

    def _clear(self) -> None:
        """Clear the dependency builder."""
        self.next_node_id = 0
        self.dep_graph = nx.DiGraph()
        self.last_disp_of_modes = defaultdict(lambda: None)
        self.last_op_of_modes = {}
        self.measurement_to_mode = {}

    def _apply_displacement(self, target: int) -> None:
        modes = self.dep_graph.nodes[target]["modes"]
        for mode in modes:
            prev_disp = self.last_disp_of_modes[mode]
            if prev_disp is not None:
                self.dep_graph.nodes[target]["displacements"].append((mode, prev_disp))
                self.last_disp_of_modes[mode] = None

    def _apply_dependency(self, target: int) -> None:
        modes = self.dep_graph.nodes[target]["modes"]
        for mode in modes:
            if mode in self.last_op_of_modes:
                self.dep_graph.add_edge(self.last_op_of_modes[mode], target)

            self.last_op_of_modes[mode] = target

    def _apply_feedforward(self, target: int) -> None:
        op: GraphOp = self.dep_graph.nodes[target]["op"]

        params = list(op.parameters)
        params.extend(self.dep_graph.nodes[target]["displacements"])
        for p in params:
            if not isinstance(p, FeedForward):
                continue
            if not isinstance(p.variable, ModeMeasuredVariable):
                msg = "The type of parameters of operation does not match."
                raise TypeError(msg)
            if p.variable.mode not in self.measurement_to_mode:
                msg = "The mode of ModeMeasuredVariable does not match."
                raise ValueError(msg)
            measurement_ind = self.measurement_to_mode[p.variable.mode]

            for n in self.dep_graph.predecessors(target):
                # avoid deadlock
                if nx.has_path(self.dep_graph, measurement_ind, n):
                    continue
                self.dep_graph.add_edge(n, measurement_ind)

            self.dep_graph.add_edge(measurement_ind, target)
            self.dep_graph.nodes[target]["has_nlff"] = True

    def _add_op_node(self, op: GraphOp, modes: list[int]) -> None:
        self.dep_graph.add_node(self.next_node_id)
        self.dep_graph.nodes[self.next_node_id]["op"] = op
        self.dep_graph.nodes[self.next_node_id]["modes"] = modes
        self.dep_graph.nodes[self.next_node_id]["displacements"] = []

        if isinstance(op, Measurement):
            self.measurement_to_mode[modes[0]] = self.next_node_id

        self._apply_displacement(self.next_node_id)
        self._apply_dependency(self.next_node_id)
        self._apply_feedforward(self.next_node_id)

        self.next_node_id += 1

    def _add_initialization(self, mode: int, theta: float = math.pi / 2) -> None:
        init = gops.Initialization((DEFAULT_COORDINATE), theta=theta, initialized_modes=(BLANK_MODE, mode))
        self._add_op_node(init, [mode])

    def _add_displacement(self, disp: cops.Displacement) -> None:
        mode = disp.opnd().get_ids()[0]

        prev_disp = self.last_disp_of_modes[mode]
        if prev_disp is None:
            self.last_disp_of_modes[mode] = _convert_cg_param(disp.x), _convert_cg_param(disp.p)
        else:
            x, p = prev_disp
            self.last_disp_of_modes[mode] = (
                _add_c_disp_to_g_disp(disp.x, x),
                _add_c_disp_to_g_disp(disp.p, p),
            )

    def _add_displacement_from_graph(self, mode: int, params: tuple[GraphOpParam, GraphOpParam]) -> None:
        prev_disp = self.last_disp_of_modes[mode]
        if prev_disp is None:
            self.last_disp_of_modes[mode] = params
        else:
            self.last_disp_of_modes[mode] = (
                _add_g_disps(params[0], prev_disp[0]),
                _add_g_disps(params[1], prev_disp[1]),
            )

    def _reset_op(self, op: GraphOp) -> GraphOp:
        op.macronode = DEFAULT_COORDINATE
        op.swap = False
        op.displacement_k_minus_1 = (0.0, 0.0)
        op.displacement_k_minus_n = (0.0, 0.0)
        return op

    def from_circuit(self, circuit: CircuitRepr) -> nx.DiGraph:
        self._clear()

        for mode, initial_state in enumerate(circuit.initial_states):
            if isinstance(initial_state, HardwareConstrainedSqueezedState):
                # The `theta` argument is the measurement angle.
                # `initial_state.phi` is the squeezing angle of the initialized mode.
                self._add_initialization(mode, theta=initial_state.phi - pi / 2)
            else:
                self._add_initialization(mode)

        for op_in_circ in circuit:
            if isinstance(op_in_circ, cops.Displacement):
                self._add_displacement(op_in_circ)
                continue

            modes = op_in_circ.opnd().get_ids()
            for op_in_graph in convert_op(op_in_circ, (DEFAULT_COORDINATE)):
                self._add_op_node(op_in_graph, modes)

        return self.dep_graph

    def from_graph(self, graph: GraphRepr) -> nx.DiGraph:
        self._clear()
        modes = [BLANK_MODE] * (graph.n_local_macronodes + 1)
        for w in range(graph.n_steps):
            for h in range(graph.n_local_macronodes):
                left, up = modes[h], modes[-1]
                down, right = up, left
                op = graph.get_operation(h, w)
                if left != BLANK_MODE:
                    self._add_displacement_from_graph(left, op.displacement_k_minus_n)
                if up != BLANK_MODE:
                    self._add_displacement_from_graph(up, op.displacement_k_minus_1)
                if op.type() == PbOperation.OPERATION_TYPE_MEASUREMENT:
                    down, right = BLANK_MODE, BLANK_MODE
                elif op.type() == PbOperation.OPERATION_TYPE_INITIALIZATION:
                    down, right = op.initialized_modes
                elif graph.is_swap_macronode(h, w):
                    down, right = left, up
                if op.type() != PbOperation.OPERATION_TYPE_WIRING:
                    op = self._reset_op(op)
                    self._add_op_node(op, list({left, right, up, down} - {BLANK_MODE}))

                modes[h] = right
                modes[-1] = down

        return self.dep_graph
