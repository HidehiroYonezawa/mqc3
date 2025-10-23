"""Test the beamsearch embedder."""

# pyright: reportUnusedExpression=false
from collections import Counter, defaultdict

import networkx as nx
import pytest

from mqc3.circuit.ops import intrinsic
from mqc3.circuit.program import CircuitRepr
from mqc3.feedforward import feedforward
from mqc3.graph import ops
from mqc3.graph.embed.beamsearch import BeamSearchEmbedder, BeamSearchEmbedSettings
from mqc3.graph.embed.dep_dag import DependencyDAG


def run_and_verify_embedding(embedder: BeamSearchEmbedder, dep_dag: DependencyDAG) -> None:  # noqa: PLR0914
    """Runs the embedder and verifies the output against the source DAG."""
    embedded_graph = embedder.embed(dep_dag)
    dag = dep_dag.dag

    dag_ops = {op_idx: data["op"] for op_idx, data in dag.nodes(data=True)}
    expected_op_counts = Counter(type(op) for op in dag_ops.values())

    placed_ops = [op for op in embedded_graph.operations.values() if not isinstance(op, ops.Wiring)]
    actual_op_counts = Counter(type(op) for op in placed_ops)
    assert len(placed_ops) == len(dag.nodes)
    assert actual_op_counts == expected_op_counts

    mode_to_op_indices = defaultdict(list)
    for op_idx, data in dag.nodes(data=True):
        for mode in data["modes"]:
            mode_to_op_indices[mode].append(op_idx)

    topo_order = {op_idx: i for i, op_idx in enumerate(nx.topological_sort(dag))}
    expected_mode_flows = {}
    for mode, op_indices in mode_to_op_indices.items():
        sorted_indices = sorted(op_indices, key=lambda idx: topo_order[idx])
        expected_mode_flows[mode] = [type(dag_ops[idx]) for idx in sorted_indices]

    for mode_index, expected_flow in expected_mode_flows.items():
        actual_ops = embedded_graph.calc_mode_operations(mode_index)
        actual_flow = [type(op) for op in actual_ops]
        assert actual_flow == expected_flow

    # Verify feedforward distance
    ff_nodes = {idx: data for idx, data in dag.nodes(data=True) if data.get("has_nlff")}
    if ff_nodes:
        ff_op_from_dag = next(iter(ff_nodes.values()))["op"]
        predecessors = list(dag.predecessors(next(iter(ff_nodes.keys()))))
        meas_op_from_dag = next(
            dag.nodes[p]["op"] for p in predecessors if isinstance(dag.nodes[p]["op"], ops.Measurement)
        )

        measurement_op = next(op for op in placed_ops if op is meas_op_from_dag)
        feedforward_op = next(op for op in placed_ops if op is ff_op_from_dag)

        index_meas = embedded_graph.get_index(*measurement_op.macronode)
        index_ff = embedded_graph.get_index(*feedforward_op.macronode)
        distance = index_ff - index_meas

        min_dist, max_dist = embedder._settings.feedforward_distance  # noqa: SLF001
        assert min_dist <= distance <= max_dist


@pytest.fixture
def example_dag() -> DependencyDAG:
    """Fixture for a simple DependencyDAG."""
    c = CircuitRepr("test_simple_circuit")
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(1) | intrinsic.PhaseRotation(0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)
    c.Q(0) | intrinsic.Measurement(0.0)
    c.Q(1) | intrinsic.Measurement(0.0)
    return DependencyDAG(c)


@pytest.fixture
def example_dag_arbitrary() -> DependencyDAG:
    """Fixture for a DependencyDAG with Arbitrary operation."""
    c = CircuitRepr("test_arbitrary")
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)
    c.Q(3) | intrinsic.ShearXInvariant(0.0)
    c.Q(2, 4) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(2, 3) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(4) | intrinsic.PhaseRotation(0.0)
    c.Q(1, 4) | intrinsic.ControlledZ(0.0)
    c.Q(0) | intrinsic.Arbitrary(0.0, 0.0, 0.0)
    for i in range(5):
        c.Q(i) | intrinsic.Measurement(0.0)
    return DependencyDAG(c)


@pytest.fixture
def example_dag_ff() -> DependencyDAG:
    """Fixture for a DependencyDAG with a feedforward dependency."""
    c = CircuitRepr("test_feedforward_simple")
    x = c.Q(0) | intrinsic.Measurement(0.0)

    @feedforward
    def f(v: float) -> float:
        return v

    c.Q(1) | intrinsic.PhaseRotation(f(x))
    return DependencyDAG(c)


@pytest.fixture
def beamsearch_embed_settings() -> BeamSearchEmbedSettings:
    """Fixture for BeamSearchEmbedSettings."""
    return BeamSearchEmbedSettings(n_local_macronodes=5, feedforward_distance=(5, 20))


@pytest.fixture
def beamsearch_embedder(beamsearch_embed_settings: BeamSearchEmbedSettings) -> BeamSearchEmbedder:
    """Fixture for the BeamSearchEmbedder instance."""
    return BeamSearchEmbedder(settings=beamsearch_embed_settings)


def test_embedding_dag(
    beamsearch_embedder: BeamSearchEmbedder,
    example_dag: DependencyDAG,
) -> None:
    """Tests embedding a DependencyDAG."""
    run_and_verify_embedding(beamsearch_embedder, example_dag)


def test_embedding_dag_arbitrary(
    beamsearch_embedder: BeamSearchEmbedder,
    example_dag_arbitrary: DependencyDAG,
) -> None:
    """Tests embedding a DependencyDAG with arbitrary operation."""
    run_and_verify_embedding(beamsearch_embedder, example_dag_arbitrary)


def test_embedding_feedforward_circuit(
    beamsearch_embedder: BeamSearchEmbedder,
    example_dag_ff: DependencyDAG,
) -> None:
    """Tests embedding a DependencyDAG with feedforward dependencies."""
    run_and_verify_embedding(beamsearch_embedder, example_dag_ff)
