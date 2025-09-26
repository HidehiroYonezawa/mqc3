"""Test macro angle."""

from math import pi

import pytest

from mqc3.feedforward import FeedForward
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
from mqc3.graph.program import GraphRepr, Wiring
from mqc3.machinery.macronode_angle import (
    MacronodeAngle,
    from_45_squeeze,
    from_arbitrary_first,
    from_arbitrary_second,
    from_beam_splitter,
    from_controlled_z,
    from_graph_operation,
    from_initialization,
    from_manual,
    from_measurement,
    from_phase_rotation,
    from_shear_p_invariant,
    from_shear_x_invariant,
    from_squeeze,
    from_two_mode_shear,
    from_wiring,
)
from mqc3.math import equiv_mod_pi
from mqc3.pb.mqc3_cloud.program.v1.machinery_pb2 import MachineryRepresentation as PbMachineryRepr

EPSILON = 1e-10


AngleTuple = tuple[float, float, float, float]


def assert_macronode_angle(ma: MacronodeAngle, angles: AngleTuple) -> None:
    assert not isinstance(ma.theta_a, FeedForward)
    assert not isinstance(ma.theta_b, FeedForward)
    assert not isinstance(ma.theta_c, FeedForward)
    assert not isinstance(ma.theta_d, FeedForward)
    equiv_mod_pi(ma.theta_a, angles[0])
    equiv_mod_pi(ma.theta_b, angles[1])
    equiv_mod_pi(ma.theta_c, angles[2])
    equiv_mod_pi(ma.theta_d, angles[3])


def test_macronode_angle() -> None:
    assert MacronodeAngle(1, 2, 3, 4)
    assert MacronodeAngle(1, 2, 1, 2)
    assert MacronodeAngle(1, 2, 2, 1)
    assert MacronodeAngle(1, 1, 1, 1)
    assert MacronodeAngle(1 + pi, 1 - 2 * pi, 1 - pi, 1 - pi)
    with pytest.raises(ValueError, match="The four angles must satisfy the following relation modulo pi"):
        assert MacronodeAngle(1, 1, 2, 3)
    with pytest.raises(ValueError, match="The four angles must satisfy the following relation modulo pi"):
        assert MacronodeAngle(1, 2, 3, 3)
    with pytest.raises(ValueError, match="The four angles must satisfy the following relation modulo pi"):
        assert MacronodeAngle(1 + pi, 1 + pi, 2, 3)
    with pytest.raises(ValueError, match="The four angles must satisfy the following relation modulo pi"):
        assert MacronodeAngle(1, 2, 3 + pi, 3 + pi)
    with pytest.raises(ValueError, match="The four angles must satisfy the following relation modulo pi"):
        assert MacronodeAngle(1 - 5 * pi, 1, 2, 3)
    with pytest.raises(ValueError, match="The four angles must satisfy the following relation modulo pi"):
        assert MacronodeAngle(1, 2, 3, 3 - 5 * pi)
    proto = MacronodeAngle(1, 2, 3, 4).proto()
    isinstance(proto, PbMachineryRepr.MacronodeAngle)
    isinstance(MacronodeAngle.construct_from_proto(proto), MacronodeAngle)


@pytest.mark.parametrize(argnames=("theta"), argvalues=[0.0, pi / 4, pi / 2])
def test_from_measurement(theta: float) -> None:
    expected = (theta, theta, theta, theta)

    ma = from_measurement(theta)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(Measurement((0, 0), theta))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)

    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(argnames=("theta"), argvalues=[0.0, pi / 4, pi / 2])
def test_from_initialization(theta: float) -> None:
    expected = (theta, theta, theta, theta)

    ma = from_initialization(theta)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(Initialization((0, 0), theta, (0, 0)))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("phi", "swap", "expected"),
    argvalues=[
        (0.0, False, (0.0, pi / 2.0, 0.0, pi / 2.0)),
        (0.0, True, (pi / 2.0, 0.0, 0.0, pi / 2.0)),
        (-pi, False, (-pi / 2.0, 0.0, -pi / 2.0, 0.0)),
        (-pi, True, (0.0, -pi / 2.0, -pi / 2.0, 0.0)),
    ],
)
def test_from_phase_rotation(phi: float, *, swap: bool, expected: tuple[float, float, float, float]) -> None:
    ma = from_phase_rotation(phi, swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(PhaseRotation((0, 0), phi, swap=swap))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("k", "swap", "expected"),
    argvalues=[
        (1.0, False, (pi / 4.0, pi / 2.0, pi / 4.0, pi / 2.0)),
        (1.0, True, (pi / 2.0, pi / 4.0, pi / 4.0, pi / 2.0)),
    ],
)
def test_from_shear_x_invariant(k: float, *, swap: bool, expected: AngleTuple) -> None:
    ma = from_shear_x_invariant(k, swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(ShearXInvariant((0, 0), k, swap=swap))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("e", "swap", "expected"),
    argvalues=[
        (-1.0, False, (0, 3.0 * pi / 4.0, 0, 3.0 * pi / 4.0)),
        (-1.0, True, (3.0 * pi / 4.0, 0, 0, 3.0 * pi / 4.0)),
    ],
)
def test_from_shear_p_invariant(e: float, *, swap: bool, expected: AngleTuple) -> None:
    ma = from_shear_p_invariant(e, swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(ShearPInvariant((0, 0), e, swap=swap))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("a", "b", "swap", "expected"),
    argvalues=[
        (0.0, 2.0, False, (-pi / 4.0, pi / 2.0, pi / 4.0, pi / 2.0)),
        (0.0, 2.0, True, (pi / 2.0, -pi / 4.0, pi / 4.0, pi / 2.0)),
    ],
)
def test_from_two_mode_shear(a: float, b: float, *, swap: bool, expected: AngleTuple) -> None:
    ma = from_two_mode_shear(a, b, swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(TwoModeShear((0, 0), a, b, swap=swap))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("theta", "swap", "expected"),
    argvalues=[
        (pi / 2, False, (-pi / 2, pi / 2, -pi / 2, pi / 2)),
        (pi / 2, True, (pi / 2, -pi / 2, -pi / 2, pi / 2)),
        (pi / 2.0, False, (-pi / 2.0, pi / 2.0, -pi / 2.0, pi / 2.0)),
        (pi / 2.0, True, (pi / 2.0, -pi / 2.0, -pi / 2.0, pi / 2.0)),
    ],
)
def test_from_squeeze(theta: float, *, swap: bool, expected: AngleTuple) -> None:
    ma = from_squeeze(theta, swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(Squeezing((0, 0), theta, swap=swap))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("theta", "swap", "expected"),
    argvalues=[
        (pi / 2, False, (-pi / 4.0, 3 * pi / 4.0, -pi / 4.0, 3 * pi / 4.0)),
        (pi / 2, True, (3 * pi / 4.0, -pi / 4.0, -pi / 4.0, 3 * pi / 4.0)),
        (pi / 4.0, False, (0.0, pi / 2.0, 0.0, pi / 2.0)),
        (pi / 4.0, True, (pi / 2.0, 0.0, 0.0, pi / 2.0)),
    ],
)
def test_from_45_squeezing(theta: float, *, swap: bool, expected: AngleTuple) -> None:
    ma = from_45_squeeze(theta, swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(Squeezing45((0, 0), theta, swap=swap))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("a", "b", "lam", "swap", "expected"),
    argvalues=[
        (
            0.0,
            0.0,
            0.0,
            (False, False),
            (
                (-pi / 4.0, pi / 4.0, -pi / 4.0, pi / 4.0),
                (pi / 4.0, -pi / 4.0, pi / 4.0, -pi / 4.0),
            ),
        ),
        (
            0.0,
            0.0,
            0.0,
            (False, True),
            (
                (-pi / 4.0, pi / 4.0, -pi / 4.0, pi / 4.0),
                (-pi / 4.0, pi / 4.0, pi / 4.0, -pi / 4.0),
            ),
        ),
        (
            0.0,
            0.0,
            0.0,
            (True, False),
            (
                (pi / 4.0, -pi / 4.0, -pi / 4.0, pi / 4.0),
                (pi / 4.0, -pi / 4.0, pi / 4.0, -pi / 4.0),
            ),
        ),
        (
            0.0,
            0.0,
            0.0,
            (True, True),
            (
                (pi / 4.0, -pi / 4.0, -pi / 4.0, pi / 4.0),
                (-pi / 4.0, pi / 4.0, pi / 4.0, -pi / 4.0),
            ),
        ),
    ],
)
def test_from_arbitrary(
    a: float,
    b: float,
    lam: float,
    swap: tuple[bool, bool],
    expected: tuple[AngleTuple, AngleTuple],
) -> None:
    ma0 = from_arbitrary_first(a, b, lam, swap=swap[0])
    ma1 = from_arbitrary_second(a, b, lam, swap=swap[1])
    assert_macronode_angle(ma0, expected[0])
    assert_macronode_angle(ma1, expected[1])

    # mode move in 1st macronode: (from up, to left)
    if swap[0]:
        graph = GraphRepr(2, 3)
        graph.place_operation(Initialization(macronode=(0, 0), theta=0.0, initialized_modes=(BLANK_MODE, 0)))
        graph.place_operation(Wiring((0, 1), swap=True))

        # mode move in 2nd macronode:
        # if swap[1]: (from right, to left)
        # if not swap[1]: (from right, to down)
        arb_fst = ArbitraryFirst((1, 1), a, b, lam, swap=swap[0])
        arb_snd = ArbitrarySecond((1, 2), a, b, lam, swap=swap[1])

    # mode move in 1st macronode: (from right, to left)
    else:
        graph = GraphRepr(1, 3)
        graph.place_operation(Initialization(macronode=(0, 0), theta=0.0, initialized_modes=(BLANK_MODE, 0)))

        # mode move in 2nd macronode:
        # if swap[1]: (from right, to down)
        # if not swap[1]: (from right, to left)
        arb_fst = ArbitraryFirst((0, 1), a, b, lam, swap=swap[0])
        arb_snd = ArbitrarySecond((0, 2), a, b, lam, swap=swap[1])

    graph.place_operation(arb_fst)
    graph.place_operation(arb_snd)
    ma_from_op_0 = from_graph_operation(arb_fst, graph)
    ma_from_op_1 = from_graph_operation(arb_snd, graph)

    assert isinstance(ma_from_op_0, MacronodeAngle)
    assert isinstance(ma_from_op_1, MacronodeAngle)
    assert_macronode_angle(ma_from_op_0, expected[0])
    assert_macronode_angle(ma_from_op_1, expected[1])

    # mode move in 1st macronode: (from right, to down)
    if swap[0]:
        graph = GraphRepr(2, 3)
        graph.place_operation(Initialization(macronode=(0, 0), theta=0.0, initialized_modes=(BLANK_MODE, 0)))

        # mode move in 2nd macronode:
        # if swap[1]: (from up, to left)
        # if not swap[1]: (from up, to down)
        arb_fst = ArbitraryFirst((0, 1), a, b, lam, swap=swap[0])
        arb_snd = ArbitrarySecond((1, 1), a, b, lam, swap=swap[1])

    # mode move in 1st macronode: (from up, to down)
    else:
        graph = GraphRepr(3, 2)
        graph.place_operation(Initialization(macronode=(0, 0), theta=0.0, initialized_modes=(BLANK_MODE, 0)))
        graph.place_operation(Wiring((0, 1), swap=True))

        # mode move in 2nd macronode:
        # if swap[1]: (from up, to left)
        # if not swap[1]: (from up, to down)
        arb_fst = ArbitraryFirst((1, 1), a, b, lam, swap=swap[0])
        arb_snd = ArbitrarySecond((2, 1), a, b, lam, swap=swap[1])

    graph.place_operation(arb_fst)
    graph.place_operation(arb_snd)
    ma_from_op_0 = from_graph_operation(arb_fst, graph)
    ma_from_op_1 = from_graph_operation(arb_snd, graph)
    assert isinstance(ma_from_op_0, MacronodeAngle)
    assert isinstance(ma_from_op_1, MacronodeAngle)
    assert_macronode_angle(ma_from_op_0, expected[0])
    assert_macronode_angle(ma_from_op_1, expected[1])


@pytest.mark.parametrize(
    argnames=("g", "swap", "expected"),
    argvalues=[
        (2.0, False, (-pi / 4.0, pi / 2.0, pi / 4.0, pi / 2.0)),
        (2.0, True, (pi / 2.0, -pi / 4.0, pi / 4.0, pi / 2.0)),
    ],
)
def test_from_controlled_z(g: float, *, swap: bool, expected: AngleTuple) -> None:
    ma = from_controlled_z(g, swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(ControlledZ((0, 0), g, swap=swap))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("sqrt_r", "theta_rel", "swap", "expected"),
    argvalues=[
        (0.0, 0.0, False, (pi / 4, 3 * pi / 4, -pi / 4, pi / 4)),
        (0.0, 0.0, True, (3 * pi / 4, pi / 4, -pi / 4, pi / 4)),
        (1.0, pi / 2, False, (pi / 4, 3 * pi / 4, pi / 4, 3 * pi / 4)),
        (1.0, pi / 2, True, (3 * pi / 4, pi / 4, pi / 4, 3 * pi / 4)),
    ],
)
def test_from_beam_splitter(sqrt_r: float, theta_rel: float, *, swap: bool, expected: AngleTuple) -> None:
    ma = from_beam_splitter(sqrt_r, theta_rel, swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(BeamSplitter((0, 0), sqrt_r, theta_rel, swap=swap))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("abcd", "swap", "expected"),
    argvalues=[
        ([1.0, 2.0, 3.0, 4.0], False, (1.0, 2.0, 3.0, 4.0)),
        ([1.0, 2.0, 3.0, 4.0], True, (2.0, 1.0, 3.0, 4.0)),
    ],
)
def test_from_manual(abcd: list[float], *, swap: bool, expected: AngleTuple) -> None:
    ma = from_manual(*abcd, swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(Manual((0, 0), *abcd, swap=swap))  # type: ignore  # noqa: PGH003
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)


@pytest.mark.parametrize(
    argnames=("swap", "expected"),
    argvalues=[
        (False, (0.0, pi / 2.0, 0.0, pi / 2.0)),
        (True, (pi / 2.0, 0.0, 0.0, pi / 2.0)),
    ],
)
def test_from_wiring(*, swap: bool, expected: AngleTuple) -> None:
    ma = from_wiring(swap=swap)
    assert_macronode_angle(ma, expected)

    graph = GraphRepr(1, 1)
    graph.place_operation(Wiring((0, 0), swap=swap))
    ma_from_op = from_graph_operation(graph.get_operation(0, 0), graph)
    assert isinstance(ma_from_op, MacronodeAngle)
    assert_macronode_angle(ma_from_op, expected)
