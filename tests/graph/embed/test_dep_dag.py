"""Test dependency DAG."""

# pyright: reportUnusedExpression=false
import networkx as nx
import pytest

from mqc3.circuit.ops import intrinsic as cops
from mqc3.circuit.program import CircuitRepr
from mqc3.feedforward import feedforward
from mqc3.graph import ops
from mqc3.graph.embed.dep_dag import DependencyDAG
from mqc3.graph.ops import ControlledZ, Initialization, Measurement, PhaseRotation
from mqc3.graph.program import GraphRepr


@pytest.fixture
def example_dag() -> DependencyDAG:
    """Fixture to create a simple DependencyDAG without feedforward."""
    c = CircuitRepr("test_simple_circuit")
    c.Q(0) | cops.PhaseRotation(0.0)
    c.Q(1) | cops.PhaseRotation(0.0)
    c.Q(0, 1) | cops.ControlledZ(0.0)
    c.Q(0) | cops.Measurement(0.0)
    c.Q(1) | cops.Measurement(0.0)
    return DependencyDAG(c)


@pytest.fixture
def example_dag_ff() -> DependencyDAG:
    """Fixture for a simple DependencyDAG with a feedforward dependency."""
    c = CircuitRepr("test_feedforward_simple")
    x = c.Q(0) | cops.Measurement(0.0)

    @feedforward
    def f(v: float) -> float:
        return v

    c.Q(1) | cops.PhaseRotation(f(x))
    return DependencyDAG(c)


@pytest.fixture
def simple_graph() -> GraphRepr:
    """A simple GraphRepr for testing DAG construction from a graph."""
    g = GraphRepr(n_local_macronodes=2, n_steps=3)
    g.place_operation(ops.Initialization((0, 0), initialized_modes=(0, 1), theta=0.0))
    g.place_operation(ops.PhaseRotation((0, 1), phi=0.0, swap=False))
    g.place_operation(ops.Measurement((1, 2), theta=0.3))
    return g


def test_circuit_dependencies(example_dag: DependencyDAG):
    """Tests the dependencies of a standard circuit."""
    dag = example_dag.dag
    assert len(dag.nodes) == 7

    init_nodes = {d["modes"][0]: n for n, d in dag.nodes(data=True) if isinstance(d["op"], Initialization)}
    pr_nodes = {d["modes"][0]: n for n, d in dag.nodes(data=True) if isinstance(d["op"], PhaseRotation)}
    cz_node = next(n for n, d in dag.nodes(data=True) if isinstance(d["op"], ControlledZ))
    meas_nodes = {d["modes"][0]: n for n, d in dag.nodes(data=True) if isinstance(d["op"], Measurement)}

    assert set(dag.predecessors(init_nodes[0])) == set()
    assert set(dag.predecessors(init_nodes[1])) == set()
    assert set(dag.predecessors(pr_nodes[0])) == {init_nodes[0]}
    assert set(dag.predecessors(pr_nodes[1])) == {init_nodes[1]}
    assert set(dag.predecessors(cz_node)) == {pr_nodes[0], pr_nodes[1]}
    assert set(dag.predecessors(meas_nodes[0])) == {cz_node}
    assert set(dag.predecessors(meas_nodes[1])) == {cz_node}


def test_feedforward_dependencies(example_dag_ff: DependencyDAG):
    """Tests feedforward dependencies."""
    dag = example_dag_ff.dag
    assert len(dag.nodes) == 4

    init_nodes = {d["modes"][0]: n for n, d in dag.nodes(data=True) if isinstance(d["op"], Initialization)}
    meas_node = next(n for n, d in dag.nodes(data=True) if isinstance(d["op"], Measurement) and d["modes"] == [0])
    pr_node = next(n for n, d in dag.nodes(data=True) if isinstance(d["op"], PhaseRotation) and d["modes"] == [1])

    assert set(dag.predecessors(meas_node)) == {init_nodes[0], init_nodes[1]}
    assert init_nodes[1] in set(dag.predecessors(pr_node))
    assert meas_node in set(dag.predecessors(pr_node))
    assert dag.nodes[pr_node].get("has_nlff") is True


def test_from_graph_conversion(simple_graph: GraphRepr):
    """Tests DAG conversion from a GraphRepr."""
    dag = DependencyDAG(simple_graph).dag
    assert len(dag.nodes) == 3

    nodes = list(nx.topological_sort(dag))
    init_node, pr_node, meas_node = nodes[0], nodes[1], nodes[2]

    assert isinstance(dag.nodes[init_node]["op"], ops.Initialization)
    assert isinstance(dag.nodes[pr_node]["op"], ops.PhaseRotation)
    assert isinstance(dag.nodes[meas_node]["op"], ops.Measurement)

    assert set(dag.predecessors(pr_node)) == {init_node}
    assert set(dag.predecessors(meas_node)) == {pr_node}
