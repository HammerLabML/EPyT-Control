"""
This module contains some helper functions.
"""
import numpy as np


def is_mat_spsd(mat: np.ndarray) -> bool:
    """
    Checks if a given matrix is symmetric positive semi-definite.

    Parameters
    ----------
    mat : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Matrix to be check.

    Returns
    -------
    `bool`
        True if 'mat' is symmetric positive semi-definite, False otherwise.
    """
    return np.array_equal(mat, mat.T) and np.all(np.linalg.eigvals(mat) >= 0)


def is_mat_spd(mat: np.ndarray) -> bool:
    """
    Checks if a given matrix is symmetric positive definite.

    Parameters
    ----------
    mat : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Matrix to be check.

    Returns
    -------
    `bool`
        True if 'mat' is symmetric positive definite, False otherwise.
    """
    return np.array_equal(mat, mat.T) and np.all(np.linalg.eigvals(mat) > 0)
