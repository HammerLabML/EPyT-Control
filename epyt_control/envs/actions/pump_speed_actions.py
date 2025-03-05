"""
This module contains a class for modeling pump speed action spaces.
"""
from typing import Optional
from epyt_flow.gym import ScenarioControlEnv
from gymnasium.spaces import Space, Box

from .actions import Action


class PumpSpeedAction(Action):
    """
    Action for controlling the speed of a pump.

    Parameters
    ----------
    pump_id : `str`
        ID of the pump.
    speed_lower_bound : `float`, optional
        Lower bound on the pump speed.

        The default is zero.
    speed_upper_bound : `float`, optional
        Upper bound on the pump speed.
        If None, no upper bound restriction for the agent.

        The default is None.
    """
    def __init__(self, pump_id: str, speed_lower_bound: float = 0.,
                 speed_upper_bound: Optional[float] = None, **kwds):
        if not isinstance(pump_id, str):
            raise TypeError("'pump_id' must be an instance of 'str' " +
                            f"but not of '{type(pump_id)}'")
        if not isinstance(speed_lower_bound, float):
            raise TypeError("'speed_lower_bound' must be an instance of 'float' " +
                            f"but not of '{type(speed_lower_bound)}'")
        if speed_lower_bound < 0:
            raise ValueError("'speed_lower_bound' can not be negative!")
        if speed_upper_bound is not None:
            if not isinstance(speed_upper_bound, float):
                raise TypeError("'speed_upper_bound' must be an instance of 'float' " +
                                f"but not of '{type(speed_upper_bound)}'")
            if speed_upper_bound < 0:
                raise ValueError("'speed_upper_bound' can not be negative!")

        self._pump_id = pump_id
        self._speed_lower_bound = speed_lower_bound
        self._speed_upper_bound = speed_upper_bound \
            if speed_upper_bound is not None else float("inf")

        super().__init__(**kwds)

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self._pump_id == other.pump_id and \
            self._speed_lower_bound == other.speed_lower_bound and\
            self._speed_upper_bound == other.speed_upper_bound

    def __str__(self) -> str:
        return super().__str__() + f"Pump ID: {self._pump_id} " +\
            f"Speed lower bound: {self._speed_lower_bound} " +\
            f"Speed upper bound: {self._speed_upper_bound}"

    @property
    def pump_id(self) -> str:
        """
        Return the ID of the pump.

        Returns
        -------
        `str`
            ID of the pump.
        """
        return self._pump_id

    @property
    def speed_lower_bound(self) -> float:
        """
        Returns the lower bound of the pump speed that can be set by the agent.

        Returns
        -------
        `float`
            Lower bound on the pump speed.
        """
        return self._speed_lower_bound

    @property
    def speed_upper_bound(self) -> float:
        """
        Returns the upper bound of the pump speed that can be set by the agent.

        Returns
        -------
        `float`
            Upper bound on the pump speed.
        """
        return self._speed_upper_bound

    def to_gym_action_space(self) -> Space:
        return Box(low=self._speed_lower_bound, high=self._speed_upper_bound)

    def apply(self, env: ScenarioControlEnv, action_value: float) -> None:
        env.set_pump_speed(self._pump_id, action_value)
