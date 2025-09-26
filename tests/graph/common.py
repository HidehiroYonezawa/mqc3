"""Common module for graph tests."""

from mqc3.graph import GraphRepr, Wiring
from mqc3.graph.constant import BLANK_MODE
from mqc3.graph.ops import (
    BeamSplitter,
    ControlledZ,
    Initialization,
    Measurement,
    PhaseRotation,
    ShearXInvariant,
)


def make_sample_graph() -> GraphRepr:
    g = GraphRepr(5, 6)
    for i in range(5):
        g.place_operation(Initialization(macronode=(i, 0), theta=0.0, initialized_modes=(BLANK_MODE, i)))

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
    return g
