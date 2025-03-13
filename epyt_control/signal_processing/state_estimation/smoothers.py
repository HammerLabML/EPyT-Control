"""
This module contains several smoothing methods for improving filters.
"""
from typing import Optional, Callable
import numpy as np

from .kalman_filters import KalmanFilter, TimeVaryingKalmanFilter


class RauchTungStriebelSmoother(KalmanFilter):
    """
    Implementation of the Rauch-Tung-Striebel Kalman filter smoother.

    Parameters
    ----------
    time_window_length : `int`
        Length of the time window which is considered for smoothing.
    state_dim : `int`
        Dimensionality of states.
    obs_dim : `int`
        Dimensionality of observations.
    init_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Initial state.
    measurement_func : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Measurement function -- i.e. matrix that is converting a state into an observation.
    state_transition_func : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        State transition function -- i.e. matrix moving from a given state to the next state.
    init_state_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the initial state uncertainty.
        If None, the identity matrix will be used.

        The default is None.
    measurement_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the measurement/observation uncertainty.
        If None, the identity matrix will be used.

        The default is None.
    system_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the system uncertainty.
        If None, the identity matrix will be used.

        The default is None.
    """
    def __init__(self, time_window_length: int, state_dim: int, obs_dim: int,
                 init_state: np.ndarray, measurement_func: np.ndarray,
                 state_transition_func: np.ndarray,
                 init_state_uncertainty_cov: Optional[np.ndarray],
                 measurement_uncertainty_cov: Optional[np.ndarray],
                 system_uncertainty_cov: Optional[np.ndarray]) -> None:
        if not isinstance(time_window_length, int):
            raise TypeError("'time_window_length' must be an instance of 'int' " +
                            f"but not of '{type(time_window_length)}'")
        if time_window_length <= 0:
            raise ValueError("'time_window_length' must be positive")

        self._time_window_length = time_window_length

        super().__init__(state_dim=state_dim, obs_dim=obs_dim, init_state=init_state,
                         measurement_func=measurement_func,
                         state_transition_func=state_transition_func,
                         init_state_uncertainty_cov=init_state_uncertainty_cov,
                         measurement_uncertainty_cov=measurement_uncertainty_cov,
                         system_uncertainty_cov=system_uncertainty_cov)

    @property
    def time_window_length(self) -> int:
        """
        Returns the length of the time window.

        Returns
        -------
        `int`
            Time window length.
        """
        return self._time_window_length

    def __eq__(self, other):
        return super().__eq__(other) and self._time_window_length == other.time_window_length

    def __str__(self):
        return super().__str__() + f" time_window_length: {self._time_window_length}"

    def step(self, observation: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Predicts the current state (incl. it's uncertainty) based on a given
        time window of observations.
        Also, updates all other internal states of the Kalman filter.

        Parameters
        ----------
        observation : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Time window of observations.

        Returns
        -------
        tuple[list[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], list[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]]
            Lists of predicted system states and uncertainty covariance matrices.
        """
        if not isinstance(observation, np.ndarray):
            raise TypeError("'observation' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(observation)}'")
        if observation.shape != (self._time_window_length, self._obs_dim):
            raise ValueError("'observation' must be of shap (time_window_length, obs_dim) -- " +
                             f"i.e. {(self._time_window_length, self._obs_dim)}. " +
                             f"But found {observation.shape}")

        # Forward pass
        X = [], P = []
        for i in range(self._time_window_length):
            x, cov = super().step(observation[i, :].flatten())
            X.append(x)
            P.append(cov)

        # Backward pass
        for i in range(self._time_window_length-2, -1, -1):
            C = self._F @ P[i] @ self._F.T + self._Q
            K = P[i, :] @ self._F.T @ np.linalg.inv(C)
            X[i] += K @ (X[i + 1] - (self._F @ X[i]))
            P[i] += K @ (P[i + 1] - C) @ K.T

        self._x = X[-1]
        self._P = P[-1]

        return X, P


class TimeVaryingRauchTungStriebelSmoother(TimeVaryingKalmanFilter):
    """
    Implementation of the time varying Rauch-Tung-Striebel Kalman filter smoother.

    Parameters
    ----------
    time_window_length : `int`
        Length of the time window which is considered for smoothing.
    state_dim : `int`
        Dimensionality of states.
    obs_dim : `int`
        Dimensionality of observations.
    init_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Initial state.
    measurement_func : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
        Measurement function -- i.e. matrix that is converting a state into an observation.
    state_transition_func : Callable[[int], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]
        Function mapping time (integer) to the time dependent state transition function --
        i.e. matrix moving from a given state to the next state.
    init_state_uncertainty_cov : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_, optional
        Covariance matrix of the initial state uncertainty.
        If None, the identity matrix will be used.

        The default is None.
    measurement_uncertainty_cov : Callable[[int], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], optional
        Function mapping time (integer) to the time dependent covariance matrix of the
        measurement/observation uncertainty.
        If None, the identity matrix will be used in all time steps.

        The default is None.
    system_uncertainty_cov : Callable[[int], `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], optional
        Function mapping time (integer) to the time dependent covariance matrix of the
        system uncertainty.
        If None, the identity matrix will be used in all time steps.

        The default is None.
    """
    def __init__(self, time_window_length: int, state_dim: int,
                 obs_dim: int, init_state: np.ndarray,
                 measurement_func:  np.ndarray,
                 state_transition_func: Callable[[int], np.ndarray],
                 init_state_uncertainty_cov: Optional[np.ndarray],
                 measurement_uncertainty_cov: Optional[Callable[[int], np.ndarray]],
                 system_uncertainty_cov: Optional[Callable[[int], np.ndarray]]) -> None:
        if not isinstance(time_window_length, int):
            raise TypeError("'time_window_length' must be an instance of 'int' " +
                            f"but not of '{type(time_window_length)}'")
        if time_window_length <= 0:
            raise ValueError("'time_window_length' must be positive")

        self._time_window_length = time_window_length

        super().__init__(state_dim=state_dim, obs_dim=obs_dim, init_state=init_state,
                         measurement_func=measurement_func,
                         state_transition_func=state_transition_func,
                         init_state_uncertainty_cov=init_state_uncertainty_cov,
                         measurement_uncertainty_cov=measurement_uncertainty_cov,
                         system_uncertainty_cov=system_uncertainty_cov)

    @property
    def time_window_length(self) -> int:
        """
        Returns the length of the time window.

        Returns
        -------
        `int`
            Time window length.
        """
        return self._time_window_length

    def __eq__(self, other):
        return super().__eq__(other) and self._time_window_length == other.time_window_length

    def __str__(self):
        return super().__str__() + f" time_window_length: {self._time_window_length}"

    def step(self, observation: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Predicts the current state (incl. it's uncertainty) based on a given
        time window of observations.
        Also, updates all other internal states of the Kalman filter.

        Parameters
        ----------
        observation : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Time window of observations.

        Returns
        -------
        tuple[list[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_], list[`numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_]]
            List of predicted system states and uncertainty covariance matrices.
        """
        if not isinstance(observation, np.ndarray):
            raise TypeError("'observation' must be an instance of 'numpy.ndarray' " +
                            f"but not of '{type(observation)}'")
        if observation.shape != (self._time_window_length, self._obs_dim):
            raise ValueError("'observation' must be of shap (time_window_length, obs_dim) -- " +
                             f"i.e. {(self._time_window_length, self._obs_dim)}. " +
                             f"But found {observation.shape}")

        # Forward pass
        X = [], P = []
        for i in range(self._time_window_length):
            x, cov = super().step(observation[i, :].flatten())
            X.append(x)
            P.append(cov)

        # Backward pass
        t = self._t - 1
        for i in range(self._time_window_length-2, -1, -1):
            F = self._get_state_transition_func(t)
            Q = self._get_system_uncertainty_cov(t)
            t -= 1

            C = F @ P[i] @ F.T + Q
            K = P[i, :] @ F.T @ np.linalg.inv(C)
            X[i] += K @ (X[i + 1] - (F @ X[i]))
            P[i] += K @ (P[i + 1] - C) @ K.T

        self._x = X[-1]
        self._P = P[-1]

        return X, P
