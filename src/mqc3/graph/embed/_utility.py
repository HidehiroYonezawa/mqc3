import mqc3.circuit.ops.intrinsic as cops
import mqc3.graph.ops as gops
from mqc3.circuit.program import CircOpParam, CircuitRepr
from mqc3.circuit.program import Operation as CircOp
from mqc3.feedforward import FeedForward
from mqc3.graph.ops import GraphOpParam, ModeMeasuredVariable
from mqc3.graph.program import Operation as GraphOp


def _is_displacement(op: CircOp) -> bool:
    return op.name() == "intrinsic.displacement"


def count_n_ops_except_displacement(circuit: CircuitRepr) -> int:
    return sum(not _is_displacement(op) for op in circuit)


def op_indices_except_displacement(circuit: CircuitRepr) -> list[int]:
    return [i for i, op in enumerate(circuit) if not _is_displacement(op)]


def _convert_cg_param(param: CircOpParam) -> GraphOpParam:
    if not isinstance(param, FeedForward):
        return param

    op = param.variable.get_from_operation()
    if not op.opnd().get_ids():
        msg = "Cannot convert feedforward with empty IDs."
        raise ValueError(msg)

    mode = op.opnd().get_ids()[0]

    ff = ModeMeasuredVariable(mode)
    return param.func(ff)


def convert_intrinsic_op(cop: cops.Intrinsic, coord: tuple[int, int]) -> list[GraphOp]:  # noqa: C901, PLR0911
    """Convert operation object to a new operation object in the graph representation.

    Args:
        cop (Operation): Operation object in the circuit representation.
        coord (tuple[int, int]): Coordinate of the macronode to apply or start the operation
            in the graph representation.

    Raises:
        RuntimeError: The input operation is not supported

    Returns:
        GOperation: Operation object in the graph representation.
    """
    if isinstance(cop, cops.Measurement):
        return [gops.Measurement(coord, _convert_cg_param(cop.theta))]
    if isinstance(cop, cops.PhaseRotation):
        return [gops.PhaseRotation(coord, _convert_cg_param(cop.phi), swap=False)]
    if isinstance(cop, cops.ShearXInvariant):
        return [gops.ShearXInvariant(coord, _convert_cg_param(cop.kappa), swap=False)]
    if isinstance(cop, cops.ShearPInvariant):
        return [gops.ShearPInvariant(coord, _convert_cg_param(cop.eta), swap=False)]
    if isinstance(cop, cops.Squeezing):
        return [gops.Squeezing(coord, _convert_cg_param(cop.theta), swap=False)]
    if isinstance(cop, cops.Squeezing45):
        return [gops.Squeezing45(coord, _convert_cg_param(cop.theta), swap=False)]
    if isinstance(cop, cops.Arbitrary):
        arb_first = gops.ArbitraryFirst(
            coord,
            _convert_cg_param(cop.alpha),
            _convert_cg_param(cop.beta),
            _convert_cg_param(cop.lam),
            swap=False,
        )
        arb_second = gops.ArbitrarySecond(
            coord,
            _convert_cg_param(cop.alpha),
            _convert_cg_param(cop.beta),
            _convert_cg_param(cop.lam),
            swap=False,
        )
        return [arb_first, arb_second]
    if isinstance(cop, cops.ControlledZ):
        return [gops.ControlledZ(coord, _convert_cg_param(cop.g), swap=False)]
    if isinstance(cop, cops.BeamSplitter):
        return [gops.BeamSplitter(coord, _convert_cg_param(cop.sqrt_r), _convert_cg_param(cop.theta_rel), swap=False)]
    if isinstance(cop, cops.TwoModeShear):
        return [gops.TwoModeShear(coord, _convert_cg_param(cop.a), _convert_cg_param(cop.b), swap=False)]
    if isinstance(cop, cops.Manual):
        return [
            gops.Manual(
                coord,
                _convert_cg_param(cop.theta_a),
                _convert_cg_param(cop.theta_b),
                _convert_cg_param(cop.theta_c),
                _convert_cg_param(cop.theta_d),
                swap=False,
            )
        ]
    msg = "This operation is not supported"
    raise RuntimeError(msg)


def convert_op(cop: CircOp, coord: tuple[int, int]) -> list[GraphOp]:
    """Convert operation object to a new operation object in the graph representation.

    Args:
        cop (Operation): Operation object in the circuit representation.
        coord (tuple[int, int]): Coordinate of the macronode to apply or start the operation
            in the graph representation.

    Raises:
        TypeError: One of the return values of to_intrinsic_ops is not an intrinsic operation.

    Returns:
        GOperation: Operation object in the graph representation.
    """
    gop_list = []
    for intrinsic_op in cop.to_intrinsic_ops():
        if not isinstance(intrinsic_op, cops.Intrinsic):
            msg = "One of the return values of to_intrinsic_ops is not an intrinsic operation."
            raise TypeError(msg)
        gop_list.extend(convert_intrinsic_op(intrinsic_op, coord))
    return gop_list
