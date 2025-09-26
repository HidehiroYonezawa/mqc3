"""Test for feedforward."""

import pytest

from mqc3.feedforward import FeedForward, FeedForwardFunction, Variable, feedforward


class V(Variable):
    def get_from_operation(self) -> None:  # noqa: D102
        return None


def test_construct():
    def f(x: float) -> float:
        return x + 1

    ff_func = FeedForwardFunction()

    ff_func._add_func(f)  # noqa: SLF001

    assert len(ff_func._func_def_list) == 1  # noqa: SLF001

    # call with float
    assert ff_func(1) == 2
    assert ff_func(2) == 3

    # call with Variable
    v = V()
    ff = ff_func(v)
    assert isinstance(ff, FeedForward)


def test_ff_decorator():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    assert isinstance(f, FeedForwardFunction)
    assert f(1) == 2

    v = V()
    ff = f(v)
    assert isinstance(ff, FeedForward)

    @feedforward
    def g(x: float) -> float:
        return x * 2

    assert isinstance(g, FeedForwardFunction)
    ff = g(f(v))
    assert isinstance(ff, FeedForward)
    assert len(ff._func._func_def_list) == 2  # noqa: SLF001
    assert ff.func(1) == 4
    assert ff.func(2) == 6


def test_invalid_func():
    with pytest.raises(ValueError, match="Function must have a single argument"):

        @feedforward  # type: ignore  # noqa: PGH003
        def f(x: float, y: float) -> float:  # noqa: FURB118
            return x + y


def test_duplicate_func():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    v = V()
    ff = f(f(v))
    assert isinstance(ff, FeedForward)
    assert len(ff._func._func_def_list) == 2  # noqa: SLF001
    assert ff.func(1) == 3
    assert ff.func(2) == 4


def test_convert_between_proto_and_ff():
    @feedforward
    def f(x: float) -> float:
        return x + 1

    @feedforward
    def g(x: float) -> float:
        return x * 2

    # single function
    v = V()
    ff = f(v)
    assert isinstance(ff, FeedForward)
    proto = ff._func.proto()  # noqa: SLF001
    reconstructed = FeedForwardFunction.construct_from_proto(proto)
    assert len(reconstructed._func_def_list) == 1  # noqa: SLF001
    assert reconstructed(1) == 2
    assert reconstructed(2) == 3

    # multiple functions
    ff = g(f(v))
    assert isinstance(ff, FeedForward)
    proto = ff._func.proto()  # noqa: SLF001
    reconstructed = FeedForwardFunction.construct_from_proto(proto)
    assert len(reconstructed._func_def_list) == 2  # noqa: SLF001
    assert reconstructed(1) == 4

    # duplicate functions
    ff = f(f(v))
    assert isinstance(ff, FeedForward)
    proto = ff._func.proto()  # noqa: SLF001
    reconstructed = FeedForwardFunction.construct_from_proto(proto)
    assert len(reconstructed._func_def_list) == 2  # noqa: SLF001
    assert reconstructed(1) == 3


def test_feedforward_binary_op_and_import_from_math():
    @feedforward
    def f(x: float) -> float:
        from math import sin  # noqa: PLC0415

        a = x + 1
        a2 = a - 2
        a3 = a2 * 3
        a4 = a3 / 4
        a5 = a4 // 5
        a6 = a5 % 6
        a7 = int(a6) << 7
        a8 = a7 >> 8
        a9 = a8 & 9
        a10 = a9 ^ 10
        a11 = a10 | 11
        a12 = a11**12

        return sin(a12)


def test_feedforward_builtin_function_call():
    @feedforward
    def f(x: float) -> float:
        a = abs(x)
        a = a * bool(2)  # noqa: PLR6104
        a = a * complex(3, 4)  # noqa: PLR6104
        a, _ = divmod(a.real, 5)
        a = float(a)
        a = int(a)
        a = pow(a, 3)
        a = round(a)

        return a  # noqa: RET504


def test_feedforward_multiple_assign():
    @feedforward
    def f(x: float) -> float:
        a = b = x**2
        return a + b


def test_feedforward_not_allowed_import():
    expected_msg = (
        "Feedforward functions can only include return statements, assignments, "
        "augmented assignments, import-from statements, and expression statements."
    )
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f_import_math(x: float) -> float:
            import math  # noqa: PLC0415

            return math.exp(x)

    from math import sin  # noqa: PLC0415

    expected_msg = (
        "Only the following functions can be called: abs, bool, complex, divmod, float, int, pow, round, "
        "and functions imported from the math module."
    )
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f_use_outside_sin(x: float) -> float:
            return sin(x) + 1

    expected_msg = "Only the built-in math module is allowed for import."
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward  # type: ignore  # noqa: PGH003
        def f(x: float):  # noqa: ANN202, ARG001
            from os import environ  # noqa: PLC0415

            return environ


def test_feedforward_assign_to_list():
    expected_msg = "Assignment targets must be either local variables or unpacked tuples."
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float) -> float:
            [p, q] = divmod(x, 2)
            return p * q


def test_feedforward_use_nonlocal_variable():
    a = 10

    expected_msg = "Names must be defined within the local scope and cannot refer to external definitions."
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float) -> float:
            return x + a


def test_feedforward_multiple_return_values():
    expected_msg = "The return value must be a single value and cannot be a tuple or None."
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward  # type: ignore  # noqa: PGH003
        def f(x: float):  # noqa: ANN202
            return x, x**2


def test_feedforward_last_stmt_is_not_return():
    expected_msg = "The function definition must end with a return statement."
    with pytest.raises(ValueError, match=expected_msg):

        @feedforward  # noqa: RET503
        def f(x: float) -> float:
            return x**2
            a = x  # noqa: F841


def test_feedforward_with_not_allowed_statement():
    expected_msg = (
        "Feedforward functions can only include return statements, assignments, "
        "augmented assignments, import-from statements, and expression statements."
    )

    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float) -> float:
            def g(y: float) -> float:
                return y**2

            return g(x) + 1

    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float) -> float:
            if x > 0:
                return x**2
            return x**3

    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f_for(x: float) -> float:  # noqa: ARG001
            ret = 1
            for i in range(1, 10):
                ret *= i

            return ret

    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f_while(x: float) -> float:
            ret = 1
            while x > 0:
                ret *= x
                x -= 1

            return ret


def test_feedforward_assign_by_undefined_variable():
    expected_msg = "Names must be defined within the local scope and cannot refer to external definitions."

    with pytest.raises(ValueError, match=expected_msg):

        @feedforward
        def f(x: float) -> float:
            a = a  # type: ignore # noqa: F821, PLW0127, PGH003
            return a * x
