"""Math."""

from math import isclose, pi


def equiv_mod_pi(a: float, b: float, abs_tol: float = 1e-3) -> bool:
    """Check whether the input values equivalent to each other modulo pi.

    Args:
        a (float): The first input value.
        b (float): The second input value.
        abs_tol (float): Tolerable absolute error.

    Returns:
        bool: Whether the input values equivalent to each other modulo pi.
    """
    return isclose(abs(a - b) % pi, 0, abs_tol=abs_tol) or isclose(abs(a - b) % pi, pi, abs_tol=abs_tol)
