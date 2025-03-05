"""
This module contains classes for modeling actuator state action spaces --
i.e. controlling pump and valve states.
"""
from copy import deepcopy
from epyt_flow.gym import ScenarioControlEnv
from epyt_flow.simulation.events import ActuatorConstants
from gymnasium.spaces import Space, Discrete

from .actions import Action


class ActuatorStateAction(Action):
    """
    Base class of an actuator state action.

    Parameters
    ----------
    state_space : `list[int]`
        List of possible actions that can be taken by the agent.
    """
    def __init__(self, state_space: list[int], **kwds):
        if not isinstance(state_space, list):
            raise TypeError("'state_space' must be an instance of 'list[int]' " +
                            f"but not of '{type(state_space)}'")
        if any(not isinstance(state, int) for state in state_space):
            raise TypeError("All states in 'state_space' must be an instance of 'int'")
        self._action_space = state_space

        super().__init__(**kwds)

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self._action_space == other.action_space

    def __str__(self) -> str:
        return super().__str__() + f"Action space: {self._action_space}"

    @property
    def action_space(self) -> list[int]:
        """
        Returns the list of possible actions that an agent can take.

        Returns
        -------
        `list[int]`
            List of possible actions.
        """
        return deepcopy(self._action_space)

    def to_gym_action_space(self) -> Space:
        return Discrete(len(self._action_space))


class ValveStateAction(ActuatorStateAction):
    """
    Action for controlling the state of a valve.

    Parameters
    ----------
    valve_id : `str`
        ID of the valve.
    """
    def __init__(self, valve_id: str, **kwds):
        self._valve_id = valve_id

        super().__init__(state_space=[ActuatorConstants.EN_OPEN, ActuatorConstants.EN_CLOSED],
                         **kwds)

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self.valve_id == other.valve_id

    def __str__(self) -> str:
        return super().__str__() + f"Valve ID: {self._valve_id}"

    @property
    def valve_id(self) -> str:
        """
        Returns the ID of the valve.

        Returns
        -------
        `str`
            ID of the valve.
        """
        return self._valve_id

    def apply(self, env: ScenarioControlEnv, action_value: int) -> None:
        env.set_valve_status(self._valve_id, action_value)


class PumpStateAction(ActuatorStateAction):
    """
    Action for controling the state of a pump.

    Parameters
    ----------
    pump_id : `str`
        ID of the pump.
    """
    def __init__(self, pump_id: str, **kwds):
        self._pump_id = pump_id

        super().__init__(state_space=[ActuatorConstants.EN_OPEN, ActuatorConstants.EN_CLOSED],
                         **kwds)

    def __eq__(self, other) -> bool:
        return super().__eq__(other) and self._pump_id == other.pump_id

    def __str__(self) -> str:
        return super().__str__() + f"Pump ID: {self._pump_id}"

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

    def apply(self, env: ScenarioControlEnv, action_value) -> None:
        env.set_pump_status(self._pump_id, action_value)
