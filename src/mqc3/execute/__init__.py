"""Module for executing quantum circuits using a specified client.

This module provides the :func:`~mqc3.execute.execute` function and the :class:`~mqc3.execute.ExecutionResult` class
to run quantum circuits via different client backends.
"""

from mqc3.execute._execute import execute
from mqc3.execute._result import ExecutionResult

__all__ = ["ExecutionResult", "execute"]
