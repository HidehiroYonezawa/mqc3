"""Test operations of circuit representation."""

# pyright: reportUnusedExpression=false

from math import pi
from sys import float_info

import pytest

from mqc3.circuit import CircuitRepr, QuMode
from mqc3.circuit.ops import intrinsic, std
from mqc3.circuit.ops.intrinsic import BeamSplitter, Manual, Squeezing, Squeezing45
from mqc3.circuit.program import MeasuredVariable
from mqc3.feedforward import FeedForward, feedforward
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitOperation as PbOperation
from mqc3.pb.mqc3_cloud.program.v1.circuit_pb2 import CircuitRepresentation as PbCircuitRepr


def test_program():
    c = CircuitRepr("test_qumode")
    c.Q(5) | intrinsic.Displacement(0.0, 0.0)
    assert c.Q(5).modes[0].id == 5
    assert str(c.Q(5).modes[0]) == "5"
    c.Q(7, 8) | intrinsic.ControlledZ(0.0)
    for qumode in c.Q(7, 8):
        assert isinstance(qumode, QuMode)
    assert str(c.Q(7, 8)) == "[QuMode(id=7), QuMode(id=8)]"
    assert repr(c.Q(7, 8)) == "[QuMode(id=7), QuMode(id=8)]"


def test_program_error():
    c = CircuitRepr("test_qumode")
    op = intrinsic.Squeezing(1)
    with pytest.raises(RuntimeError, match=r"Operand is not set."):
        op.opnd()
    assert str(op) == "[intrinsic.squeezing] [1]"
    c.Q(4) | op
    assert str(op) == "[intrinsic.squeezing] [1] [QuMode(id=4)]"
    with pytest.raises(RuntimeError, match="The number of modes for operation"):
        c.Q(5, 6) | op
    with pytest.raises(RuntimeError, match="The argument of Q"):
        c._push_one([1, 2, 3], "a")  # pyright: ignore[reportArgumentType] # noqa: SLF001
    c._push([1, 2, 3], [1, 2, 3, 4])  # noqa: SLF001
    assert isinstance(c.proto(), PbCircuitRepr)
    assert len(c) == 1
    assert str(c) == "[intrinsic.squeezing] [1] [QuMode(id=4)]"
    assert repr(c) == "[intrinsic.squeezing] [1] [QuMode(id=4)]"


def test_all_intrinsic():
    c = CircuitRepr("test_all_intrinsic")
    c.Q(0) | intrinsic.Displacement(0.0, 0.0)
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(0) | intrinsic.ShearXInvariant(0.0)
    c.Q(0) | intrinsic.ShearPInvariant(0.0)
    c.Q(0) | intrinsic.Squeezing(1.0)
    c.Q(0) | intrinsic.Squeezing45(1.0)
    c.Q(0) | intrinsic.Arbitrary(0.0, 0.0, 0.0)
    c.Q(0, 1) | intrinsic.ControlledZ(0.0)
    c.Q(0, 1) | intrinsic.BeamSplitter(0.0, 0.0)
    c.Q(0, 1) | intrinsic.TwoModeShear(0.0, 0.0)
    c.Q(0, 1) | intrinsic.Manual(0.0, 1.0, 0.0, 1.0)
    c.Q(0) | intrinsic.Measurement(0.0)

    assert c.name == "test_all_intrinsic"
    assert c.n_modes == 2
    assert c.n_operations == 12
    assert isinstance(c.get_operation(i=0), intrinsic.Displacement)
    assert isinstance(c.get_operation(i=1), intrinsic.PhaseRotation)
    assert isinstance(c.get_operation(i=2), intrinsic.ShearXInvariant)
    assert isinstance(c.get_operation(i=3), intrinsic.ShearPInvariant)
    assert isinstance(c.get_operation(i=4), intrinsic.Squeezing)
    assert isinstance(c.get_operation(i=5), intrinsic.Squeezing45)
    assert isinstance(c.get_operation(i=6), intrinsic.Arbitrary)
    assert isinstance(c.get_operation(i=7), intrinsic.ControlledZ)
    assert isinstance(c.get_operation(i=8), intrinsic.BeamSplitter)
    assert isinstance(c.get_operation(i=9), intrinsic.TwoModeShear)
    assert isinstance(c.get_operation(i=10), intrinsic.Manual)
    assert isinstance(c.get_operation(i=11), intrinsic.Measurement)

    pb_c = c.proto()
    CircuitRepr.construct_from_proto(pb_c)


def test_all_std():
    c = CircuitRepr("test_all_std")
    c.Q(0) | std.Squeezing(0.0)
    c.Q(1, 2) | std.BeamSplitter(0.0, 0.0)

    assert c.name == "test_all_std"
    assert c.n_modes == 3
    assert c.n_operations == 2

    assert isinstance(c.get_operation(i=0), std.Squeezing)
    assert c.get_operation(i=0).opnd().get_ids() == [0]
    assert isinstance(c.get_operation(i=1), std.BeamSplitter)
    assert c.get_operation(i=1).opnd().get_ids() == [1, 2]

    pb_c = c.proto()
    CircuitRepr.construct_from_proto(pb_c)

    c.convert_std_ops_to_intrinsic()
    assert c.name == "test_all_std"
    assert c.n_modes == 3
    assert c.n_operations == 6
    assert isinstance(c.get_operation(i=0), intrinsic.Arbitrary)
    assert c.get_operation(i=0).opnd().get_ids() == [0]
    assert isinstance(c.get_operation(i=1), intrinsic.PhaseRotation)
    assert c.get_operation(i=1).opnd().get_ids() == [1]
    assert isinstance(c.get_operation(i=2), intrinsic.PhaseRotation)
    assert c.get_operation(i=2).opnd().get_ids() == [2]
    assert isinstance(c.get_operation(i=3), intrinsic.Manual)
    assert c.get_operation(i=3).opnd().get_ids() == [1, 2]
    assert isinstance(c.get_operation(i=4), intrinsic.PhaseRotation)
    assert c.get_operation(i=4).opnd().get_ids() == [1]
    assert isinstance(c.get_operation(i=5), intrinsic.PhaseRotation)
    assert c.get_operation(i=5).opnd().get_ids() == [2]

    pb_c = c.proto()
    CircuitRepr.construct_from_proto(pb_c)


def test_intrinsic_squeezing_err():
    assert Squeezing(theta=1)
    assert Squeezing(theta=-1.0)
    assert Squeezing(theta=5 * pi + 1)
    assert Squeezing(theta=5 * pi - 1)
    assert Squeezing(theta=-5 * pi + 1)
    assert Squeezing(theta=-5 * pi - 1)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(theta=0)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(theta=0.0)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(theta=5 * pi)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing(theta=-5 * pi)


def test_intrinsic_squeezing45_err():
    assert Squeezing45(theta=1)
    assert Squeezing45(theta=-1.0)
    assert Squeezing(theta=5 * pi + 1)
    assert Squeezing(theta=5 * pi - 1)
    assert Squeezing(theta=-5 * pi + 1)
    assert Squeezing(theta=-5 * pi - 1)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(theta=0)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(theta=0.0)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(theta=5 * pi)
    with pytest.raises(ValueError, match="must not be an integer multiple of pi"):
        Squeezing45(theta=-5 * pi)


def test_intrinsic_beam_splitter_err():
    assert BeamSplitter(sqrt_r=0.5, theta_rel=1.0)
    assert BeamSplitter(sqrt_r=float_info.epsilon, theta_rel=1.0)
    assert BeamSplitter(sqrt_r=1.0 - float_info.epsilon, theta_rel=1.0)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(sqrt_r=1.0 + float_info.epsilon, theta_rel=1.0)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(sqrt_r=-float_info.epsilon, theta_rel=1.0)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(sqrt_r=2, theta_rel=1.0)
    with pytest.raises(ValueError, match="must be in the range"):
        BeamSplitter(sqrt_r=-1, theta_rel=1.0)


def test_intrinsic_manual_err():
    assert Manual(1, 2, 3, 4)
    assert Manual(1, 2, 1, 2)
    assert Manual(1, 2, 2, 1)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 1, 1, 1)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 1, 2, 3)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 2, 3, 3)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1.0, 1, 2.0, 3.0)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1.0, 2.0, 3.0, 3.0)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1 + pi, 1 + pi, 2, 3)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 2, 3 + pi, 3 + pi)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1 - 5 * pi, 1, 2, 3)
    with pytest.raises(ValueError, match="must not be equal to"):
        assert Manual(1, 2, 3, 3 - 5 * pi)


def test_intrinsic_proto():  # noqa:  PLR0914, PLR0915
    c1 = CircuitRepr("test_intrinsic_proto_1")
    op1 = intrinsic.Measurement(1)
    c1.Q(0) | op1
    assert isinstance(op1.proto(), list)
    assert len(op1.proto()) == 1
    assert isinstance(op1.proto()[0], PbOperation)
    assert op1.type() == PbOperation.OPERATION_TYPE_MEASUREMENT
    assert op1.n_macronodes() == 1

    c2 = CircuitRepr("test_intrinsic_proto_2")
    op2 = intrinsic.Displacement(1, 1)
    c2.Q(0) | op2
    assert isinstance(op2.proto(), list)
    assert len(op2.proto()) == 1
    assert isinstance(op2.proto()[0], PbOperation)
    assert op2.type() == PbOperation.OPERATION_TYPE_DISPLACEMENT
    assert op2.n_macronodes() == 0

    c3 = CircuitRepr("test_intrinsic_proto_3")
    op3 = intrinsic.PhaseRotation(1)
    c3.Q(0) | op3
    assert isinstance(op3.proto(), list)
    assert len(op3.proto()) == 1
    assert isinstance(op3.proto()[0], PbOperation)
    assert op3.type() == PbOperation.OPERATION_TYPE_PHASE_ROTATION
    assert op3.n_macronodes() == 1

    c4 = CircuitRepr("test_intrinsic_proto_4")
    op4 = intrinsic.ShearXInvariant(1)
    c4.Q(0) | op4
    assert isinstance(op4.proto(), list)
    assert len(op4.proto()) == 1
    assert isinstance(op4.proto()[0], PbOperation)
    assert op4.type() == PbOperation.OPERATION_TYPE_SHEAR_X_INVARIANT
    assert op4.n_macronodes() == 1

    c5 = CircuitRepr("test_intrinsic_proto_5")
    op5 = intrinsic.ShearPInvariant(1)
    c5.Q(0) | op5
    assert isinstance(op5.proto(), list)
    assert len(op5.proto()) == 1
    assert isinstance(op5.proto()[0], PbOperation)
    assert op5.type() == PbOperation.OPERATION_TYPE_SHEAR_P_INVARIANT
    assert op5.n_macronodes() == 1

    c6 = CircuitRepr("test_intrinsic_proto_6")
    op6 = intrinsic.Squeezing(1)
    c6.Q(0) | op6
    assert isinstance(op6.proto(), list)
    assert len(op6.proto()) == 1
    assert isinstance(op6.proto()[0], PbOperation)
    assert op6.type() == PbOperation.OPERATION_TYPE_SQUEEZING
    assert op6.n_macronodes() == 1

    c7 = CircuitRepr("test_intrinsic_proto_7")
    op7 = intrinsic.Squeezing45(1)
    c7.Q(0) | op7
    assert isinstance(op7.proto(), list)
    assert len(op7.proto()) == 1
    assert isinstance(op7.proto()[0], PbOperation)
    assert op7.type() == PbOperation.OPERATION_TYPE_SQUEEZING_45
    assert op7.n_macronodes() == 1

    c8 = CircuitRepr("test_intrinsic_proto_8")
    op8 = intrinsic.Arbitrary(1, 1, 1)
    c8.Q(0) | op8
    assert isinstance(op8.proto(), list)
    assert len(op8.proto()) == 1
    assert isinstance(op8.proto()[0], PbOperation)
    assert op8.type() == PbOperation.OPERATION_TYPE_ARBITRARY
    assert op8.n_macronodes() == 2

    c9 = CircuitRepr("test_intrinsic_proto_9")
    op9 = intrinsic.ControlledZ(1)
    c9.Q(0, 1) | op9
    assert isinstance(op9.proto(), list)
    assert len(op9.proto()) == 1
    assert isinstance(op9.proto()[0], PbOperation)
    assert op9.type() == PbOperation.OPERATION_TYPE_CONTROLLED_Z
    assert op9.n_macronodes() == 1

    c10 = CircuitRepr("test_intrinsic_proto_10")
    op10 = intrinsic.BeamSplitter(1, 1)
    c10.Q(0, 1) | op10
    assert isinstance(op10.proto(), list)
    assert len(op10.proto()) == 1
    assert isinstance(op10.proto()[0], PbOperation)
    assert op10.type() == PbOperation.OPERATION_TYPE_BEAM_SPLITTER
    assert op10.n_macronodes() == 1

    c11 = CircuitRepr("test_intrinsic_proto_11")
    op11 = intrinsic.TwoModeShear(1, 1)
    c11.Q(0, 1) | op11
    assert isinstance(op11.proto(), list)
    assert len(op11.proto()) == 1
    assert isinstance(op11.proto()[0], PbOperation)
    assert op11.type() == PbOperation.OPERATION_TYPE_TWO_MODE_SHEAR
    assert op11.n_macronodes() == 1

    c12 = CircuitRepr("test_intrinsic_proto_12")
    op12 = intrinsic.Manual(1, 2, 3, 4)
    c12.Q(0, 1) | op12
    assert isinstance(op12.proto(), list)
    assert len(op12.proto()) == 1
    assert isinstance(op12.proto()[0], PbOperation)
    assert op12.type() == PbOperation.OPERATION_TYPE_MANUAL
    assert op12.n_macronodes() == 1
    assert op12.name() == "intrinsic.manual"
    assert op12.parameters() == [1, 2, 3, 4]


def test_std_squeezing_err():
    c1 = CircuitRepr("test_std_squeezing_err")
    op1 = std.Squeezing(1)
    c1.Q(0) | op1
    assert isinstance(op1.proto(), list)
    assert len(op1.proto()) == 1
    assert isinstance(op1.proto()[0], PbOperation)
    assert op1.n_macronodes() == 2

    c2 = CircuitRepr("test_std_squeezing_err")
    op2 = std.BeamSplitter(1, 2)
    with pytest.raises(ValueError, match="Operands are not set"):
        op2.to_intrinsic_ops()
    c2.Q(0, 1) | op2
    assert isinstance(op2.proto(), list)
    assert len(op2.proto()) == 5
    assert isinstance(op2.proto()[0], PbOperation)
    assert op2.n_macronodes() == 5


def test_set_non_linear_feedforward():
    c = CircuitRepr("test_set_non_linear_feedforward")
    x = c.Q(0) | intrinsic.PhaseRotation(1) | intrinsic.Measurement(0)
    assert isinstance(x, MeasuredVariable)
    assert isinstance(x.get_from_operation(), intrinsic.Measurement)

    @feedforward
    def f(x: float) -> float:
        return x + 1

    c.Q(1) | intrinsic.PhaseRotation(f(x))
    op = c.get_operation(2)
    assert isinstance(op, intrinsic.PhaseRotation)
    assert isinstance(op.phi, FeedForward)


def test_contain_feedforward_params():
    c = CircuitRepr("test_contain_feedforward_params")
    x = c.Q(0) | intrinsic.PhaseRotation(1) | intrinsic.Measurement(0)
    assert isinstance(x, MeasuredVariable)
    assert isinstance(x.get_from_operation(), intrinsic.Measurement)

    @feedforward
    def f(x: float) -> float:
        return x + 1

    c.Q(1) | intrinsic.PhaseRotation(f(x))
    c.Q(1) | intrinsic.Displacement(1, 2)

    assert c.get_operation(0).has_feedforward_param() is False
    assert c.get_operation(1).has_feedforward_param() is False
    assert c.get_operation(2).has_feedforward_param() is True
    assert c.get_operation(3).has_feedforward_param() is False


def test_repr_to_proto():
    # without non-linear feedforward
    c = CircuitRepr("test_repr_to_proto")
    c.Q(0) | intrinsic.Displacement(0.0, 0.0)
    c.Q(0) | intrinsic.PhaseRotation(0.0)
    c.Q(0) | intrinsic.ShearXInvariant(0.0)

    pb_c = c.proto()

    # with non-linear feedforward
    x = c.Q(1) | intrinsic.ShearPInvariant(0.0) | intrinsic.Measurement(0)

    @feedforward
    def f(x: float) -> float:
        return x + 1

    c.Q(2) | intrinsic.PhaseRotation(f(x))
    pb_c = c.proto()
    reconstructed = CircuitRepr.construct_from_proto(pb_c)
    op = reconstructed.get_operation(5)
    assert isinstance(op, intrinsic.PhaseRotation)
    assert isinstance(op.phi, FeedForward)
    assert op.phi._func(1) == 2  # noqa: SLF001

    pb_c = reconstructed.proto()
    reconstructed = CircuitRepr.construct_from_proto(pb_c)
