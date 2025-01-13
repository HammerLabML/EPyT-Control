"""
This module contains a base class for actions.
"""
from abc import abstractmethod
from gymnasium.spaces import Space
from epyt_flow.gym import ScenarioControlEnv


class Action():
    """
    Base class for actions.
    """
    @abstractmethod
    def to_gym_action_space(self) -> Space:
        """
        Converts this action to a
        `gymnasium.spaces.Space <https://gymnasium.farama.org/api/spaces/#gymnasium.spaces.Space>`_
        instance.

        Returns
        -------
        `gymnasium.spaces.Space <https://gymnasium.farama.org/api/spaces/#gymnasium.spaces.Space>`_
            gymnasium.spaces.Space instance.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(self, env: ScenarioControlEnv, action_value) -> None:
        """
        Applies a given action (from the action space) in a given environment.

        Parameters
        ----------
        env : `epyt_flow.gym.ScenarioControlEnv <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.gym.html#epyt_flow.gym.scenario_control_env.ScenarioControlEnv>`_
            The environment.
        action_value: Any
            The action.
        """
        raise NotImplementedError()
