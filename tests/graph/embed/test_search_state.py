"""Test search state."""

# pyright: reportUnusedExpression=false
# ruff: noqa: SLF001
import networkx as nx
import pytest

from mqc3.circuit.ops import intrinsic
from mqc3.circuit.program import CircuitRepr
from mqc3.feedforward import feedforward
from mqc3.graph.embed._search_state import SearchState  # noqa: PLC2701
from mqc3.graph.embed.dep_dag import DependencyDAG
from mqc3.graph.embed.embed import GraphEmbedSettings


@pytest.fixture
def example_dag() -> DependencyDAG:
    """Fixture to create a simple DependencyDAG without feedforward."""
    c = CircuitRepr("test_simple_circuit")
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(1) | intrinsic.PhaseRotation(0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)
    c.Q(0) | intrinsic.Measurement(0.0)
    c.Q(1) | intrinsic.Measurement(0.0)
    return DependencyDAG(c)


@pytest.fixture
def example_dag_ff() -> DependencyDAG:
    """Fixture for a simple DependencyDAG with a feedforward dependency."""
    c = CircuitRepr("test_feedforward_simple")
    x = c.Q(0) | intrinsic.Measurement(0.0)

    @feedforward
    def f(v: float) -> float:
        return v

    c.Q(1) | intrinsic.PhaseRotation(f(x))
    return DependencyDAG(c)


@pytest.fixture
def example_state(example_dag: DependencyDAG) -> SearchState:
    """Fixture for SearchState initialized with a DependencyDAG from a circuit."""
    settings = GraphEmbedSettings(n_local_macronodes=5, feedforward_distance=(5, 20))
    return SearchState(example_dag, settings)


@pytest.fixture
def example_state_ff(example_dag_ff: DependencyDAG) -> SearchState:
    """Fixture for SearchState initialized with a simple feedforward DAG."""
    settings = GraphEmbedSettings(n_local_macronodes=3, feedforward_distance=(1, 5))
    return SearchState(example_dag_ff, settings)


def test_place_initialization_operation(example_state: SearchState):
    """Tests correct placement of an Initialization operation."""
    # Operation 0 is Initialization for mode 0.
    init_op_ind = 0
    assert not example_state.is_already_placed(init_op_ind)
    example_state.place_operation(init_op_ind, swap_in_op=False)
    assert example_state.is_already_placed(init_op_ind)
    assert len(example_state._op_pos_dict) == 1


def test_place_single_mode_operation(example_state: SearchState):
    """Tests correct placement of a single-mode operation (PhaseRotation)."""
    # Operation 0 is Initialization for mode 0.
    init_op_ind = 0
    example_state.place_operation(init_op_ind, swap_in_op=False)

    # Operation 2 is the PhaseRotation on mode 0.
    single_mode_op_ind = 2
    assert not example_state.is_already_placed(single_mode_op_ind)
    example_state.place_operation(single_mode_op_ind, swap_in_op=False)
    assert example_state.is_already_placed(single_mode_op_ind)
    assert example_state.get_coord(example_state.index - 1) == example_state._op_pos_dict[single_mode_op_ind]


def test_place_two_mode_operation(example_state: SearchState):
    """Tests correct placement of a two-mode operation (ControlledZ)."""
    # Op 0: Init(0), Op 1: Init(1)
    # Op 2: PhaseRotation(0) [depends on 0]
    # Op 3: PhaseRotation(1) [depends on 1]
    # Op 4: ControlledZ(0, 1) [depends on 2, 3]
    example_state.place_operation(0, swap_in_op=False)
    example_state.place_operation(1, swap_in_op=False)
    example_state.place_operation(2, swap_in_op=False)
    example_state.place_operation(3, swap_in_op=False)

    # Operation 4 is ControlledZ on modes 0 and 1.
    two_mode_op_ind = 4
    assert not example_state.is_already_placed(two_mode_op_ind)
    example_state.place_operation(two_mode_op_ind, swap_in_op=False)
    assert example_state.is_already_placed(two_mode_op_ind)
    assert example_state.get_coord(example_state.index - 1) == example_state._op_pos_dict[two_mode_op_ind]


def test_place_operation_with_feedforward(example_state_ff: SearchState):
    """Tests placement of an operation with a feedforward dependency."""
    # In example_dag_ff:
    # Op 0: Init(0), Op 1: Init(1)
    # Op 2: Measurement(0) [depends on 0]
    # Op 3: PhaseRotation(1) [depends on 1 and 2 (feedforward)]
    example_state_ff.place_operation(0, swap_in_op=False)
    example_state_ff.place_operation(1, swap_in_op=False)

    meas_op_ind = 2
    example_state_ff.place_operation(meas_op_ind, swap_in_op=False)

    phase_rot_op_ind = 3
    example_state_ff.place_operation(phase_rot_op_ind, swap_in_op=False)

    meas_pos = example_state_ff.get_index(example_state_ff._op_pos_dict[meas_op_ind])
    rot_pos = example_state_ff.get_index(example_state_ff._op_pos_dict[phase_rot_op_ind])
    assert rot_pos >= meas_pos + example_state_ff.feedforward_distance[0]
    assert rot_pos <= meas_pos + example_state_ff.feedforward_distance[1]


def test_generate_next_states_and_is_all_done(example_state: SearchState):
    """Tests the generation of next states and the completion check."""
    assert not example_state.is_all_done()

    op_indices = list(nx.topological_sort(example_state.dep_dag.dag))
    for op_ind in op_indices:
        if not example_state.is_already_placed(op_ind):
            example_state.place_operation(op_ind, swap_in_op=False)

    assert example_state.is_all_done()

    next_states_gen = example_state.generate_next_states()
    assert list(next_states_gen) == []


def test_calc_placeable_range_concrete(example_state_ff: SearchState):
    """Tests that calc_placeable_range returns the correct concrete range."""
    example_state_ff.place_operation(0, swap_in_op=False)
    example_state_ff.place_operation(1, swap_in_op=False)
    example_state_ff.place_operation(2, swap_in_op=False)

    meas_op_ind = 2
    meas_pos_index = example_state_ff.get_index(example_state_ff._op_pos_dict[meas_op_ind])

    expected_min = meas_pos_index + example_state_ff.feedforward_distance[0]
    expected_max = meas_pos_index + example_state_ff.feedforward_distance[1]

    phase_rot_op_ind = 3
    placeable_range = example_state_ff.calc_placeable_range(phase_rot_op_ind)

    assert placeable_range is not None
    min_index, max_index = placeable_range
    assert min_index == expected_min
    assert max_index == expected_max


def test_place_operation_error_handling(example_state: SearchState):
    """Tests that placing operations with errors raises RuntimeError."""
    # Test placing an op with unresolved dependencies.
    # Operation 2 (PhaseRotation) depends on Operation 0 (Init).
    with pytest.raises(RuntimeError, match=r"This operation is not placeable currently."):
        example_state.place_operation(2, swap_in_op=False)

    # Test placing an already-placed op.
    example_state.place_operation(0, swap_in_op=False)
    with pytest.raises(RuntimeError, match="This operation has been already placed"):
        example_state.place_operation(0, swap_in_op=False)


def test_place_single_mode_operation_with_swap(example_state: SearchState):
    """Tests correct placement of a single-mode operation with a swap."""
    example_state.place_operation(0, swap_in_op=False)  # Init(0)
    op_ind = 2  # PhaseRotation on mode 0

    example_state.place_operation(op_ind, swap_in_op=True)

    assert example_state.is_already_placed(op_ind)
    op_pos = example_state._op_pos_dict[op_ind]

    # Verify that a swap was registered at the operation's position
    assert op_pos in example_state._swap_pos_set
