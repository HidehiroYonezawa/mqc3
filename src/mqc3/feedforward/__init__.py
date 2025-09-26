"""This module provides feedforward functionality.

In MQC3, **feedforward** refers to the process of updating the parameters of certain operations
in a quantum circuit based on measurement outcomes obtained during circuit execution.

Feedforward consists of two main components: **variables** and **functions**.

A *variable* represents a measurement result obtained during the execution of a quantum program.
In this SDK, variables are represented by the class :class:`~mqc3.feedforward.Variable` or its subclasses.

Feedforward functions are defined using the :class:`~mqc3.feedforward.FeedForwardFunction` class.
You can create a feedforward function by decorating a regular Python function with the
:func:`~mqc3.feedforward.feedforward` decorator.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Generic, TypeVar

from mqc3.feedforward.verification import verify_feedforward
from mqc3.pb.mqc3_cloud.common.v1.function_pb2 import PythonFunction as PbPythonFunction

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _make_ff_func_string(f: Callable[[float], float]) -> str:
    source = textwrap.dedent(inspect.getsource(f))
    parsed = ast.parse(source)
    for body in parsed.body:
        if isinstance(body, ast.FunctionDef) and body.decorator_list:
            if not isinstance(body.decorator_list[0], ast.Name) or body.decorator_list[0].id != "feedforward":
                logger.warning(
                    "Feedforward function ignore decorators. (except for ``mqc3.feedforward.feedforward``).",
                )

            body.decorator_list = []

    return ast.unparse(parsed)


def _load_func_from_str(func_str: str) -> Callable[[float], float]:
    local = {}
    exec(func_str, {}, local)  # noqa: S102
    return next(iter(local.values()))


_VarElem = TypeVar("_VarElem")


class Variable(ABC, Generic[_VarElem]):  # noqa: UP046, RUF100
    """Variable for feedforward."""

    @abstractmethod
    def get_from_operation(self) -> _VarElem:
        """Get the operation that the variable is from.

        Returns:
            _VarElem: Operation.
        """

    def __repr__(self) -> str:
        """Return a string representation of the variable."""
        return f"Var({self.get_from_operation()})"


_Var = TypeVar("_Var", bound=Variable)


class FeedForward(Generic[_Var]):  # noqa: UP046, RUF100
    """FeedForward class.

    FeedForward is a class that represents a non-linear feedforward function.
    """

    _variable: _Var
    _func: FeedForwardFunction

    def __init__(self, var: _Var) -> None:
        """Initialize a FeedForward.

        Args:
            var (Variable): Variable to apply feedforward to.
        """
        super().__init__()
        self._variable = var
        self._func = FeedForwardFunction()

    @property
    def variable(self) -> _Var:
        """Get the variable of the feedforward.

        Returns:
            _Var: Variable.
        """
        return self._variable

    @property
    def func(self) -> FeedForwardFunction:
        """Get the feedforward function.

        Returns:
            FeedForwardFunction: Feedforward function.
        """
        return self._func

    def __repr__(self) -> str:
        """Return a string representation of the feedforward."""
        return f"FeedForward({self._variable})"


class FeedForwardFunction:
    """FeedForwardFunction class."""

    _func_def_list: list[str]
    _func_cache_list: list[Callable[[float], float]]

    def __init__(self) -> None:
        """Initialize a FeedForwardFunction."""
        super().__init__()
        self._func_def_list = []
        self._func_cache_list = []

    def _append_ff_func(self, other: FeedForwardFunction) -> None:
        """Append a feedforward function to the feedforward function.

        Args:
            other (FeedForwardFunction): Feedforward function to append.
        """
        self._func_def_list += other._func_def_list  # noqa: SLF001, RUF100
        self._func_cache_list += other._func_cache_list  # noqa: SLF001, RUF100

    def _add_func(self, f: Callable[[float], float]) -> None:
        """Add a feedforward function to the feedforward function.

        Args:
            f (Callable[[float], float]): Feedforward function to add.
        """
        func_str = _make_ff_func_string(f)
        verify_feedforward(func_str)

        self._func_def_list.append(func_str)
        self._func_cache_list.append(f)

    def __call__(self, x: Variable | FeedForward | float) -> float | FeedForward:
        """Apply the feedforward function to a variable or feedforward.

        * If the input is a float, the feedforward function is applied to the float and return a float.
        * If the input is a Variable, a new FeedForward is created with the feedforward function applied
        to the variable.
        * If the input is a FeedForward, the feedforward function is applied to the feedforward.

        Args:
            x (Variable | FeedForward | float): Variable, feedforward, or float.

        Returns:
            float | FeedForward: Result of applying the feedforward function.
        """
        if isinstance(x, Variable):
            ff = FeedForward(x)
            ff._func = self  # noqa: SLF001
            return ff
        if isinstance(x, FeedForward):
            # Create a new FeedForward object with the same reference to the variable of the input FeedForward object.
            ff = deepcopy(x)
            ff._variable = x.variable  # noqa: SLF001
            ff._func._append_ff_func(self)  # noqa: SLF001
            return ff

        ret = x
        if len(self._func_cache_list) != len(self._func_def_list):
            self._construct_func_cache()

        for func in self._func_cache_list:
            ret = func(ret)

        return ret

    def _construct_func_cache(self) -> None:
        """Construct cache functions from strings which define feedforward function."""
        self._func_cache_list = []
        for func_str in self._func_def_list:
            func = _load_func_from_str(func_str)
            self._func_cache_list.append(func)

    @staticmethod
    def verify(func: Callable[[float], float]) -> None:
        """Verify that the given function is safe for feedforward.

        If the function is not safe, a ValueError or TypeError is raised.

        Args:
            func (Callable[[float], float]): Function to verify.
        """
        func_str = _make_ff_func_string(func)
        verify_feedforward(func_str)

    def proto(self) -> PbPythonFunction:  # noqa: D102
        return PbPythonFunction(code=self._func_def_list)

    @staticmethod
    def construct_from_proto(proto: PbPythonFunction) -> FeedForwardFunction:  # noqa: D102
        ff_func = FeedForwardFunction()
        for c in proto.code:
            verify_feedforward(c)
            ff_func._func_def_list.append(c)

        return ff_func


def ff_to_add_constant(value: float) -> FeedForwardFunction:
    """Create a feedforward function that adds a constant value.

    Args:
        value (float): The constant value to be added.

    Returns:
        FeedForwardFunction: Feedforward function.
    """
    ff_func = FeedForwardFunction()
    f_str = f"""
def add(x: float) -> float:
    return x + {value}
"""
    ff_func._func_def_list.append(f_str)  # noqa: SLF001
    return ff_func


def ff_to_mul_constant(value: float) -> FeedForwardFunction:
    """Create a feedforward function that multiply a constant value.

    Args:
        value (float): The constant multiplier.

    Returns:
        FeedForwardFunction: Feedforward function.
    """
    ff_func = FeedForwardFunction()
    f_str = f"""
def mul(x: float) -> float:
    return x * {value}
"""
    ff_func._func_def_list.append(f_str)  # noqa: SLF001
    return ff_func


def feedforward(f: Callable[[float], float]) -> FeedForwardFunction:
    """Decorator to create a :class:`~mqc3.feedforward.FeedForwardFunction` from a function.

    Args:
        f (Callable[[float], float]): A function to decorate (must take a float and return a float).

    Returns:
        FeedForwardFunction: Feedforward function.
    """
    ff_func = FeedForwardFunction()
    ff_func._add_func(f)  # noqa: SLF001
    return ff_func
