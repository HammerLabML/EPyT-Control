"""
This module contains implementations of different Linear Quadratic Regulator (LQR) variants.
"""
from typing import Optional
import numpy as np

from .utils import is_mat_spd, is_mat_spsd


def linear_quadratic_regulator(current_state: np.ndarray, target_state: np.ndarray,
                               state_cost_mat: np.ndarray, action_cost_mat: np.ndarray,
                               state_transition_mat: np.ndarray,
                               action_transition_mat: np.ndarray, time_horizon: int,
                               final_state_cost_mat: Optional[np.ndarray] = None
                               ) -> list[np.ndarray]:
    """
    Computes the Linear Quadratic Regulator (LQR) control solution of a given
    inite-horizon & discrete-time LQR problem.

    Parameters
    ----------
    current_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Current system state.
    state_cost_mat : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Cost matrix of states -- i.e. a s.p.s.d. matrix specifying the cost of a given state.
    action_cost_mat : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Cost matrix of actions -- i.e. a s.p.d. matrix specifying the cost of a action state.
    state_transition_mat : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        State transition matrix -- i.e. mapping a given state to the next state
        (without any action).
    action_transition_mat : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Action transition matrix -- i.e. mapping specifying the state change/influence
        of taking an action.
    time_horizon : int
        Time horizon.
    final_state_cost_mat: `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Cost matrix of the final state -- i.e. a s.p.s.d. matrix specifying the cost of the final state.
        If None, 'state_cost_mat' will be used for the final state cost.

    Returns
    -------
    list[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
        List of actions for reaching the specified target space.
    """
    if not isinstance(current_state, np.ndarray):
        raise TypeError("'current_state' must be an instance of 'numpy.ndarray' " +
                        f"but not of '{type(current_state)}'")
    if current_state.ndim != 1:
        raise ValueError("'current_state' must be a 1-dimensional array -- " +
                         f"but not of shape {current_state.shape}")
    if not isinstance(state_cost_mat, np.ndarray):
        raise TypeError("'state_cost_mat' must be an instance of 'numpy.ndarray' " +
                        f"but not of '{type(state_cost_mat)}'")
    if not is_mat_spsd(state_cost_mat):
        raise ValueError("'state_cost_mat' must be symmetric positive semi-definite")
    if state_cost_mat.ndim != 2 or not state_cost_mat.shape[0] == current_state.shape[0]:
        raise ValueError("Invalid shape of 'state_cost_mat' -- " +
                         f"expecting {(current_state.shape[0], current_state.shape[0])}")
    if not isinstance(action_cost_mat, np.ndarray):
        raise TypeError("'action_cost_mat' must be an instance of 'numpy.ndarray' " +
                        f"but not of '{type(action_cost_mat)}'")
    if not is_mat_spd(action_cost_mat):
        raise ValueError("'action_cost_mat' must be symmetric positive definite")
    if not isinstance(state_transition_mat, np.ndarray):
        raise TypeError("'state_transition_mat' must be an instance of 'numpy.ndarray' " +
                        f"but not of '{type(state_transition_mat)}'")
    if state_transition_mat.shape != (current_state.shape[0], current_state.shape[0]):
        raise ValueError("Invalid shape of 'state_transition_mat' -- " +
                         f"expecting {(current_state.shape[0], current_state.shape[0])}")
    if not isinstance(action_transition_mat, np.ndarray):
        raise TypeError("'action_transition_mat' must be an instance of 'numpy.ndarray' " +
                        f"but not of '{type(action_transition_mat)}'")
    if action_transition_mat.shape[0] != current_state.shape[0] or \
            len(action_transition_mat.shape) != 2:
        raise ValueError("Invalid shape of 'action_transition_mat' -- expecting 2-dimensional " +
                         f"matrix where the first dimension is equal to {current_state.shape[0]}")
    if not isinstance(time_horizon, int):
        raise TypeError("'time_horizon' must be an instance of 'int' " +
                        f"but not of '{type(time_horizon)}'")
    if time_horizon <= 0:
        raise ValueError("'time_horizon' must be positive")
    if final_state_cost_mat is not None:
        if not isinstance(final_state_cost_mat, np.ndarray):
            raise TypeError("'final_state_cost_mat' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(final_state_cost_mat)}'")
        if not is_mat_spsd(final_state_cost_mat):
            raise ValueError("'final_state_cost_mat' must be symmetric positive semi-definite")
        if final_state_cost_mat.ndim != 2 or \
                not final_state_cost_mat.shape[0] == current_state.shape[0]:
            raise ValueError("Invalid shape of 'final_state_cost_mat' -- " +
                             f"expecting {(current_state.shape[0], current_state.shape[0])}")
    else:
        final_state_cost_mat = state_cost_mat

    P = [None] * (time_horizon + 1)
    P[time_horizon] = final_state_cost_mat

    for t in range(time_horizon, 0, -1):
        P[t-1] = state_cost_mat + state_transition_mat.T @ P[t] @ state_transition_mat - \
            (state_transition_mat.T @ P[t] @ action_transition_mat) @ \
            np.linalg.pinv(action_cost_mat + action_transition_mat.T @ P[t] @
                           action_transition_mat) @ (action_transition_mat.T @ P[t] @
                                                     state_transition_mat)

    actions = []
    x = np.copy(current_state)
    for t in range(0, time_horizon):
        K = -np.linalg.pinv(action_cost_mat + action_transition_mat.T @ P[t+1] @
                            action_transition_mat) @ \
                                action_transition_mat.T @ P[t+1] @ state_transition_mat
        u = K @ x
        x = state_transition_mat @ x + action_transition_mat @ u
        actions.append(u)

    return actions
