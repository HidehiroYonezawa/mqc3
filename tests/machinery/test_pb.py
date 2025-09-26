"""Test the serialization and deserialization of the MachineryRepr class."""

from mqc3.feedforward import feedforward
from mqc3.machinery.macronode_angle import MacronodeAngle, MeasuredVariable
from mqc3.machinery.program import MachineryRepr


def test_functions_compression():
    @feedforward
    def f1(x: float) -> float:
        return x * x

    @feedforward
    def f2(x: float) -> float:
        return x * x

    @feedforward
    def f3(x: float) -> float:
        return x * x * x

    m = MachineryRepr(
        n_local_macronodes=2,
        n_steps=1,
        readout_macronode_indices={0, 1},
    )
    x = MeasuredVariable(0, 1)
    m.set_homodyne_angle(1, MacronodeAngle(f1(x), f2(x), f3(x), f1(x)))
    m.set_displacement_k_minus_1(1, (f2(x), f3(x)))
    m.set_displacement_k_minus_n(1, (f1(x), 1.0))

    proto = m.proto()

    assert len(proto.functions) == 3
    assert len(proto.nlffs) == 7


def test_nested_functions_compression():
    @feedforward
    def f1(x: float) -> float:
        return x * x

    @feedforward
    def f2(x: float) -> float:
        return x * x * x

    @feedforward
    def g1(x: float) -> float:
        return x + 1

    @feedforward
    def g2(x: float) -> float:
        return x + 2

    m = MachineryRepr(
        n_local_macronodes=2,
        n_steps=1,
        readout_macronode_indices={0, 1},
    )
    x = MeasuredVariable(0, 1)
    m.set_homodyne_angle(1, MacronodeAngle(f1(x), f2(x), g1(f1(x)), g2(f2(x))))
    m.set_displacement_k_minus_1(1, (g2(f2(x)), f1(x)))
    m.set_displacement_k_minus_n(1, (g1(f1(x)), f2(x)))

    proto = m.proto()

    # f1, f2, g1(f1()), g2(f2())
    assert len(proto.functions) == 4
    assert len(proto.nlffs) == 8
