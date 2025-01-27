"""
This module contains an implementation of a PID controller and a
learner (i.e. tuning method) based on an evolutionary algorithm.
"""
from typing import Optional
from copy import deepcopy
import numpy as np

from ..envs.rl_env import RlEnv


class PidController():
    """
    Implementation of a Proportional-Integral-Derivative (PID) controller.

    Parameters
    ----------
    proportional_gain : `float`
        Proportional gain coefficient.
    integral_gain : `float`
        Integral gain coefficient.
    derivative_gain : `float`
        Derivative gain coefficient.
    target_value : `float`
        Target value (observed system state) which the controller is supposed to reach.
    action_lower_bound : `float`, optional
        Lower bound of the computed action.
        Smaller control outputs will be clipped.

        The default is None.
    action_upper_bound : `float`, optional
        Upper bound of the computed action.
        Control outputs exceeding this upper bound will be clipped.

        The default is None.
    """
    def __init__(self, proportional_gain: float, integral_gain: float, derivative_gain: float,
                 target_value: float, action_lower_bound: Optional[float] = None,
                 action_upper_bound: Optional[float] = None):
        if not isinstance(proportional_gain, float):
            raise TypeError("'proportional_gain' must be an instance of 'float' " +
                            f"but not of '{type(proportional_gain)}'")
        if not isinstance(integral_gain, float):
            raise TypeError("'integral_gain' must be an instance of 'float' " +
                            f"but not of '{type(integral_gain)}'")
        if not isinstance(derivative_gain, float):
            raise TypeError("'derivative_gain' must be an instance of 'float' " +
                            f"but not of '{type(derivative_gain)}'")
        if not isinstance(target_value, float):
            raise TypeError("'target_value' must be an instance of 'float' " +
                            f"but not of '{type(target_value)}'")
        if action_lower_bound is not None:
            if not isinstance(action_lower_bound, float):
                raise TypeError("'action_lower_bound' must be an instance of 'float' " +
                                f"but not of '{type(action_lower_bound)}'")
        if action_upper_bound is not None:
            if not isinstance(action_upper_bound, float):
                raise TypeError("'action_upper_bound' must be an instance of 'float' " +
                                f"but not of '{type(action_upper_bound)}'")
        if action_upper_bound is not None and action_lower_bound is not None:
            if action_lower_bound >= action_upper_bound:
                raise ValueError("'action_lower_bound' must be smaller than 'action_upper_bound'")

        self._proportional_gain = proportional_gain
        self._derivative_gain = derivative_gain
        self._integral_gain = integral_gain
        self._target_value = target_value
        self._action_lower_bound = action_lower_bound
        self._action_upper_bound = action_upper_bound

        self._last_error = 0
        self._integral = 0

    def __str__(self) -> str:
        return f"proportional_gain: {self._proportional_gain} " + \
            f"derivative_gain: {self._derivative_gain} integral_gain: {self._integral_gain} " + \
            f"target_value: {self._target_value} action_lower_bound: {self._action_lower_bound}" + \
            f" action_upper_bound: {self._action_upper_bound}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, PidController):
            raise TypeError(f"Can not compare 'PidController' to '{type(other)}'")

        return self._proportional_gain == other.proportional_gain and \
            self._derivative_gain == other.derivative_gain and \
            self._integral_gain == other.integral_gain and \
            self._target_value == other.target_value and \
            self._action_lower_bound == other.action_lower_bound and \
            self._action_upper_bound == other.action_upper_bound

    @property
    def proportional_gain(self) -> float:
        """
        Returns the proportional gain coefficient.

        Returns
        -------
        `float`
            Proportional gain coefficient.
        """
        return self._proportional_gain

    @property
    def integral_gain(self) -> float:
        """
        Returns the integral gain coefficient.

        Returns
        -------
        `float`
            Integral gain coefficient.
        """
        return self._integral_gain

    @property
    def derivative_gain(self) -> float:
        """
        Returns the derivative gain coefficient.

        Returns
        -------
        `float`
            Derivative gain coefficient.
        """
        return self._derivative_gain

    @property
    def target_value(self) -> float:
        """
        Returns the target value (observed system state) which the controller is supposed to reach.

        Returns
        -------
        `float`
            Target value (system state).
        """
        return self._target_value

    @property
    def action_lower_bound(self) -> float:
        """
        Lower bound of the computed action.
        Smaller control outputs will be clipped.

        Returns
        -------
        `float`
            Lower bound of the computed action.
        """
        return self._action_lower_bound

    @property
    def action_upper_bound(self) -> float:
        """
        Upper bound of the computed action.
        Control outputs exceeding this upper bound will be clipped.

        Returns
        -------
        `float`
            Upper bound of the computed action.
        """
        return self._action_upper_bound

    def step(self, cur_value: float) -> float:
        """
        Computes the current/next control action/signal based on the
        given observation (i.e. system state).

        Parameters
        ----------
        cur_value : `float`
            Current observation (i.e. system state).

        Returns
        -------
        `float`
            Computed action -- i.e. control signal.
        """
        error = self._target_value - cur_value
        self._integral += self._integral_gain * error

        action = self._proportional_gain * error + self._integral_gain * self._integral + \
            (self._derivative_gain * (error - self._last_error))
        if np.isnan(action):
            action = 0

        # Clip if action is outside of bounds
        if self._action_lower_bound is not None:
            action = max(action, self._action_lower_bound)
        if self._action_upper_bound is not None:
            action = min(action, self._action_upper_bound)

        self._last_error = error
        return action
