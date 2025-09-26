"""Test visualizer of graph representation."""

import pytest
from numpy import pi

from mqc3.graph import GraphRepr, Wiring
from mqc3.graph.compose import ComposeSettings, MappingInfo, compose_into_composite_graph
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    ArbitraryFirst,
    ArbitrarySecond,
    BeamSplitter,
    ControlledZ,
    Initialization,
    Manual,
    Measurement,
    PhaseRotation,
    ShearPInvariant,
    ShearXInvariant,
    Squeezing,
    Squeezing45,
    TwoModeShear,
)
from mqc3.graph.visualize import savefig

plot_on = False


def test_0node():
    g = GraphRepr(0, 0)
    with pytest.raises(ValueError, match="The original graph size must not be 0"):
        compose_into_composite_graph(
            g,
            ComposeSettings(n_shots=3, n_local_macronodes=5, max_steps=5),
        )


def test_1node():  # noqa: C901, PLR0915
    def check_map(map_info: MappingInfo, mapped_nodes: set, max_index: int) -> None:
        j = 0
        for i in range(max_index):
            if i in mapped_nodes:
                assert map_info.map[i] == (j, 0)
                j += 1
            else:
                with pytest.raises(KeyError):
                    map_info.map[i]

    g = GraphRepr(1, 1)
    g.place_operation(PhaseRotation(macronode=(0, 0), phi=0.0, swap=False))

    if plot_on:
        savefig(g, filename="fig_test_plane.pdf", show_mode_index=True)

    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=3, n_local_macronodes=5, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True)
    assert map_info.n_shots == 3
    assert n_execs_needed == 1
    check_map(map_info, mapped_nodes={0, 2, 10}, max_index=25)

    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=4, n_local_macronodes=5, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True)
    assert map_info.n_shots == 4
    assert n_execs_needed == 1
    check_map(map_info, mapped_nodes={0, 2, 10, 12}, max_index=25)

    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=5, n_local_macronodes=5, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True)
    assert map_info.n_shots == 5
    assert n_execs_needed == 1
    check_map(map_info, mapped_nodes={0, 2, 10, 12, 20}, max_index=25)

    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=8, n_local_macronodes=5, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True)
    assert map_info.n_shots == 6
    assert n_execs_needed == 2
    check_map(map_info, mapped_nodes={0, 2, 10, 12, 20, 22}, max_index=25)

    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=10, n_local_macronodes=5, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True)
    assert map_info.n_shots == 6
    assert n_execs_needed == 2
    check_map(map_info, mapped_nodes={0, 2, 10, 12, 20, 22}, max_index=25)

    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=1, n_local_macronodes=1, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True)
    assert map_info.n_shots == 1
    assert n_execs_needed == 1
    check_map(map_info, mapped_nodes={0}, max_index=5)

    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=9, n_local_macronodes=1, max_steps=5),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True)
    assert map_info.n_shots == 3
    assert n_execs_needed == 3
    check_map(map_info, mapped_nodes={0, 2, 4}, max_index=5)


def test_12nodes_no_wrap():
    g = GraphRepr(4, 3)
    g.place_operation(Initialization(macronode=(0, 0), theta=0.0, initialized_modes=(BLANK_MODE, 3)))
    g.place_operation(Initialization(macronode=(1, 0), theta=0.0, initialized_modes=(BLANK_MODE, 5)))
    g.place_operation(Initialization(macronode=(2, 0), theta=0.0, initialized_modes=(BLANK_MODE, 2)))

    if plot_on:
        savefig(g, filename="fig_test_plane.pdf", show_mode_index=True)

    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=3, n_local_macronodes=12, max_steps=10),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True)
    assert map_info.n_shots == 3
    assert n_execs_needed == 1
    assert map_info.map[0] == (0, 0)
    assert map_info.map[3] == (0, 3)
    assert map_info.map[5] == (1, 0)
    assert map_info.map[8] == (1, 3)
    assert map_info.map[20 + 4] == (0, 8)
    assert map_info.map[23 + 4] == (0, 11)
    assert map_info.map[25 + 4] == (1, 8)
    assert map_info.map[28 + 4] == (1, 11)
    assert map_info.map[40 + 8] == (2, 0)
    assert map_info.map[43 + 8] == (2, 3)
    assert map_info.map[60 + 12] == (2, 8)
    assert map_info.map[63 + 12] == (2, 11)
    for i in range(36, 48):
        with pytest.raises(KeyError):
            map_info.map[i]
    for i in range(4, 70, 12):
        with pytest.raises(KeyError):
            map_info.map[i]
    for i in range(9, 70, 12):
        with pytest.raises(KeyError):
            map_info.map[i]
    for i in range(52, 84):
        if i % 12 >= 5:
            with pytest.raises(KeyError):
                map_info.map[i]
    for i in range(84, 120):
        with pytest.raises(KeyError):
            map_info.map[i]


def test_12nodes_wrap():
    g = GraphRepr(4, 3)
    g.place_operation(Initialization(macronode=(0, 1), theta=0.0, initialized_modes=(3, 4)))
    g.place_operation(Initialization(macronode=(1, 2), theta=0.0, initialized_modes=(BLANK_MODE, 5)))
    g.place_operation(Initialization(macronode=(3, 2), theta=0.0, initialized_modes=(BLANK_MODE, 2)))

    if plot_on:
        savefig(g, filename="fig_test_plane.pdf", show_mode_index=True)

    g_composite, map_info, n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=3, n_local_macronodes=12, max_steps=10),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_composite.pdf", show_mode_index=True)
    assert map_info.n_shots == 2
    assert n_execs_needed == 2
    j = 0
    for i in range(36):
        if i % 12 < 4:
            assert map_info.map[i] == (0, j)
            j += 1
        else:
            with pytest.raises(KeyError):
                map_info.map[i]
    for i in range(36, 48):
        with pytest.raises(KeyError):
            map_info.map[i]
    j = 0
    for i in range(36):
        if i % 12 < 4:
            assert map_info.map[i + 48] == (1, j)
            j += 1
        else:
            with pytest.raises(KeyError):
                map_info.map[i + 48]
    for i in range(84, 96):
        with pytest.raises(KeyError):
            map_info.map[i]


@pytest.mark.parametrize("mode_indices", [((0, 1, 2, 3, 4)), ((3, 1, 0, 4, 2)), ((3, 1, 6, 7, 9))])
def test_mode_indices(mode_indices: tuple[int, int, int, int, int]):
    g = GraphRepr(5, 6)
    for i, i_mode in enumerate(mode_indices):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i_mode)))

    # w=1
    g.place_operation(PhaseRotation((0, 1), 0.0, swap=True))
    g.place_operation(ControlledZ((1, 1), 0.0, swap=False))
    g.place_operation(Wiring((2, 1), swap=True))
    g.place_operation(Wiring((3, 1), swap=False))
    g.place_operation(BeamSplitter((4, 1), 0.0, 0.0, swap=False))
    # w=2
    g.place_operation(Wiring((0, 2), swap=True))
    g.place_operation(Wiring((1, 2), swap=False))
    g.place_operation(Measurement((2, 2), 0.0))
    g.place_operation(ShearXInvariant((3, 2), 0.0, swap=False))
    g.place_operation(PhaseRotation((4, 2), 0.0, swap=False))
    # w=3
    g.place_operation(Wiring((0, 3), swap=True))
    g.place_operation(Wiring((1, 3), swap=False))
    g.place_operation(Wiring((2, 3), swap=False))
    g.place_operation(BeamSplitter((3, 3), 0.0, 0.0, swap=False))
    g.place_operation(Wiring((4, 3), swap=True))
    # w=4
    g.place_operation(Wiring((0, 4), swap=False))
    g.place_operation(ControlledZ((1, 4), 0.0, swap=False))
    g.place_operation(Measurement((2, 4), 0.0))
    g.place_operation(Measurement((3, 4), 0.0))
    g.place_operation(Measurement((4, 4), 0.0))
    # w=5
    g.place_operation(Wiring((0, 5), swap=False))
    g.place_operation(Measurement((1, 5), 0.0))
    g.place_operation(Wiring((2, 5), swap=False))
    g.place_operation(Wiring((3, 5), swap=False))
    g.place_operation(Wiring((4, 5), swap=False))

    if plot_on:
        savefig(g, filename="fig_test_mode_indices.pdf", show_mode_index=True)

    g_composite, _map_info, _n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=3, n_local_macronodes=30, max_steps=50),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_mode_indices_composite.pdf", scale=10, show_mode_index=True)


@pytest.mark.parametrize("mode_indices", [((0, 1)), ((1, 0)), ((9, 4))])
def test_op_params(mode_indices: tuple[int, int]):
    g = GraphRepr(6, 13)
    g.place_operation(Initialization(macronode=(0, 0), theta=pi / 2, initialized_modes=(BLANK_MODE, mode_indices[0])))
    g.place_operation(Wiring(macronode=(0, 1), swap=False))
    g.place_operation(PhaseRotation(macronode=(0, 2), phi=pi / 4, swap=False))
    g.place_operation(ShearXInvariant(macronode=(0, 3), kappa=-1, swap=False))
    g.place_operation(ShearPInvariant(macronode=(0, 4), eta=2, swap=False))
    g.place_operation(Squeezing(macronode=(0, 5), theta=3 * pi / 2, swap=False))
    g.place_operation(Squeezing45(macronode=(0, 6), theta=5 * pi / 2, swap=False))
    g.place_operation(ArbitraryFirst(macronode=(0, 7), alpha=1, beta=2, lam=3, swap=False))
    g.place_operation(ArbitrarySecond(macronode=(0, 8), alpha=1, beta=2, lam=3, swap=True))

    g.place_operation(Initialization(macronode=(1, 0), theta=pi, initialized_modes=(BLANK_MODE, mode_indices[1])))
    g.place_operation(ControlledZ(macronode=(1, 8), g=4, swap=False))

    g.place_operation(Wiring(macronode=(2, 8), swap=True))
    g.place_operation(Wiring(macronode=(1, 9), swap=True))
    g.place_operation(BeamSplitter(macronode=(2, 9), sqrt_r=0.1, theta_rel=pi / 8, swap=False))

    g.place_operation(Wiring(macronode=(3, 9), swap=True))
    g.place_operation(Wiring(macronode=(2, 10), swap=True))
    g.place_operation(TwoModeShear(macronode=(3, 10), a=10, b=20, swap=False))

    g.place_operation(Wiring(macronode=(4, 10), swap=True))
    g.place_operation(Wiring(macronode=(3, 11), swap=True))
    g.place_operation(Manual(macronode=(4, 11), theta_a=1, theta_b=2, theta_c=3, theta_d=4, swap=False))

    g.place_operation(Measurement(macronode=(5, 11), theta=0))
    g.place_operation(Measurement(macronode=(4, 12), theta=pi / 2))

    if plot_on:
        savefig(g, filename="fig_test_op_params.pdf", show_mode_index=True)

    g_composite, _map_info, _n_execs_needed = compose_into_composite_graph(
        g,
        ComposeSettings(n_shots=3, n_local_macronodes=17, max_steps=30),
    )
    if plot_on:
        savefig(g_composite, filename="fig_test_op_params_composite.pdf", scale=10, show_mode_index=True, fontsize=30)
