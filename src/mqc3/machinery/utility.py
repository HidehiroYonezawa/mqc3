"""Utility."""

from math import cos, sin

import numpy as np

from mqc3.feedforward import FeedForward
from mqc3.machinery.macronode_angle import MacronodeAngle


def construct_empty_ff_matrix() -> np.ndarray:
    """Construct an empty feedforward matrix.

    Returns:
        np.ndarray: Empty feedforward matrix.
    """
    return np.zeros((4, 4))


def calculate_ff_matrix_kp1(px_angles_k: MacronodeAngle, px_angles_k1: MacronodeAngle) -> np.ndarray:
    """Calculate the feedforward matrix for the k+1-th macronode.

    Args:
        px_angles_k (MacronodeAngle): Homodyne p-x angles in macronode k.
        px_angles_k1 (MacronodeAngle): Homodyne p-x angles in macronode k+1.

    Returns:
        np.ndarray: Feedforward matrix for the k+1-th macronode.

    Raises:
        TypeError: If the input macronode angles have feedforward.
    """
    tak, tbk, tck, tdk = px_angles_k
    if (
        isinstance(tak, FeedForward)
        or isinstance(tbk, FeedForward)
        or isinstance(tck, FeedForward)
        or isinstance(tdk, FeedForward)
    ):
        msg = "The input macronode angles have feedforward."
        raise TypeError(msg)

    tak1, tbk1, tck1, tdk1 = px_angles_k1
    if (
        isinstance(tak1, FeedForward)
        or isinstance(tbk1, FeedForward)
        or isinstance(tck1, FeedForward)
        or isinstance(tdk1, FeedForward)
    ):
        msg = "The input macronode angles have feedforward."
        raise TypeError(msg)

    denom_ab = 2 * sin(tak - tbk)
    denom_cd = 2 * sin(tck - tdk)

    if px_angles_k.is_measurable():
        matrix_k1 = (
            np.array([
                [cos(tak1 + tak)],
                [cos(tbk1 + tak)],
                [cos(tck1 + tak)],
                [cos(tdk1 + tak)],
            ])
            @ np.array([[1, -1, 1, -1]])
            / 4
        )
    else:
        matrix_k1 = -np.array([
            [sin(tak1 + tbk), sin(tak1 + tak), sin(tak1 + tdk), sin(tak1 + tck)],
            [sin(tbk1 + tbk), sin(tbk1 + tak), sin(tbk1 + tdk), sin(tbk1 + tck)],
            [sin(tck1 + tbk), sin(tck1 + tak), sin(tck1 + tdk), sin(tck1 + tck)],
            [sin(tdk1 + tbk), sin(tdk1 + tak), sin(tdk1 + tdk), sin(tdk1 + tck)],
        ])
        matrix_k1[:, :2] /= denom_ab
        matrix_k1[:, 2:] /= denom_cd

    return matrix_k1


def calculate_ff_matrix_kpn(px_angles_k: MacronodeAngle, px_angles_kn: MacronodeAngle) -> np.ndarray:
    """Calculate the feedforward matrix for the k+N-th macronode.

    Args:
        px_angles_k (MacronodeAngle): Homodyne p-x angles in macronode k.
        px_angles_kn (MacronodeAngle): Homodyne p-x angles in macronode k+N.

    Returns:
        np.ndarray: Feedforward matrix for the k+N-th macronode.

    Raises:
        TypeError: If the input macronode angles have feedforward.
    """
    tak, tbk, tck, tdk = px_angles_k
    if (
        isinstance(tak, FeedForward)
        or isinstance(tbk, FeedForward)
        or isinstance(tck, FeedForward)
        or isinstance(tdk, FeedForward)
    ):
        msg = "The input macronode angles have feedforward."
        raise TypeError(msg)

    takn, tbkn, tckn, tdkn = px_angles_kn
    if (
        isinstance(takn, FeedForward)
        or isinstance(tbkn, FeedForward)
        or isinstance(tckn, FeedForward)
        or isinstance(tdkn, FeedForward)
    ):
        msg = "The input macronode angles have feedforward."
        raise TypeError(msg)

    denom_ab = 2 * sin(tak - tbk)
    denom_cd = 2 * sin(tck - tdk)

    if px_angles_k.is_measurable():
        matrix_kn = (
            np.array([
                [cos(takn + tck)],
                [cos(tbkn + tck)],
                [-cos(tckn + tck)],
                [-cos(tdkn + tck)],
            ])
            @ np.array([[1, -1, -1, 1]])
            / 4
        )
    else:
        matrix_kn = -np.array([
            [sin(takn + tbk), sin(takn + tak), -sin(takn + tdk), -sin(takn + tck)],
            [sin(tbkn + tbk), sin(tbkn + tak), -sin(tbkn + tdk), -sin(tbkn + tck)],
            [-sin(tckn + tbk), -sin(tckn + tak), sin(tckn + tdk), sin(tckn + tck)],
            [-sin(tdkn + tbk), -sin(tdkn + tak), sin(tdkn + tdk), sin(tdkn + tck)],
        ])
        matrix_kn[:, :2] /= denom_ab
        matrix_kn[:, 2:] /= denom_cd

    return matrix_kn


def default_ff_matrices(
    px_angles_k: MacronodeAngle,
    px_angles_k1: MacronodeAngle | None = None,
    px_angles_kn: MacronodeAngle | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute default feedforward matrices in macronode k.

    Args:
        px_angles_k (MacronodeAngle): Homodyne p-x angles in macronode k.
        px_angles_k1 (MacronodeAngle | None): Homodyne p-x angles in macronode k+1. Defaults to None.
        px_angles_kn (MacronodeAngle | None): Homodyne p-x angles in macronode k+N. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pair of feedforward matrices for the k+1-th and k+N-th macronodes.
    """
    matrix_k1 = (
        calculate_ff_matrix_kp1(px_angles_k, px_angles_k1) if px_angles_k1 is not None else construct_empty_ff_matrix()
    )

    matrix_kn = (
        calculate_ff_matrix_kpn(px_angles_k, px_angles_kn) if px_angles_kn is not None else construct_empty_ff_matrix()
    )

    return matrix_k1, matrix_kn


def default_ff_matrices_all_macronodes(
    homodyne_px_angles: list[MacronodeAngle],
    n_local_macronodes: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute default feedforward matrices in all macronodes.

    Args:
        homodyne_px_angles (list[MacronodeAngle]): Homodyne p-x angles in all macronodes.
        n_local_macronodes (int): The number of macronodes per step.

    Raises:
        ValueError: If n_local_macronodes <= 0.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: Pair of list of feedforward matrices for the k+1-th
            and k+N-th macronodes. Each element of each list is feedforward matrix in the macronode.
    """
    if n_local_macronodes <= 0:
        msg = "`n_local_macronodes` must be larger than 0."
        raise ValueError(msg)
    if len(homodyne_px_angles) % n_local_macronodes != 0:
        msg = "The length of `homodyne_px_angles` must be a multiple of `n_local_macronodes`."
        raise ValueError(msg)

    matrices_k1: list[np.ndarray] = []
    matrices_kn: list[np.ndarray] = []
    n_global_macronodes = len(homodyne_px_angles)
    for i_glob_macro in range(n_global_macronodes):
        px_angles_k = homodyne_px_angles[i_glob_macro]
        plus_1_exists: bool = i_glob_macro < n_global_macronodes - 1
        plus_1_n_exist: bool = i_glob_macro < n_global_macronodes - n_local_macronodes

        px_angles_k1 = homodyne_px_angles[i_glob_macro + 1] if plus_1_exists else None
        px_angles_kn = homodyne_px_angles[i_glob_macro + n_local_macronodes] if plus_1_n_exist else None

        matrix_k1, matrix_kn = default_ff_matrices(
            px_angles_k=px_angles_k,
            px_angles_k1=px_angles_k1,
            px_angles_kn=px_angles_kn,
        )
        matrices_k1.append(matrix_k1)
        matrices_kn.append(matrix_kn)

    return matrices_k1, matrices_kn
