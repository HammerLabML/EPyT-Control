"""
This module provides functions for evaluating policies/agents/control strategies on environments.
"""
from typing import Callable, Union
import numpy as np
from epyt_flow.simulation import ScadaData
from gymnasium import Wrapper

from .envs import RlEnv


def evaluate_policy(env: Union[RlEnv, Wrapper], policy: Callable[[np.ndarray], np.ndarray],
                    n_max_iter: int = 10) -> tuple[list[float], ScadaData]:
    """
    Evaluates a given policy/agent/control strategy for a given environment --
    i.e. the policy/agent is applied to the environment and the rewards and
    `ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_
    observations over time are recorded.

    Parameters
    ----------
    env : :class:`~epyt_control.envs.rl_env.RlEnv` or `gymnasium.Wrapper <https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper>`_
        The environment.

        Note that in the case of a
        `gymnasium.Wrapper <https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper>`_
        instance, the underlying environment must be an instance of
        :class:`~epyt_control.envs.rl_env.RlEnv`.
    policy : `Callable[[numpy.ndarray], numpy.ndarray]`
        Policy/Agent/Control strategy to be evaluated.
    n_max_iter : `int`, optional
        Upper bound on the number of iterations that is used for evaluating the given policy/agent.

        The default is 1.

    Returns
    -------
    tuple[list[float], `epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_]
        Tuple of rewards over time and a
        `epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_
        instance containing the WDN states over time.
    """
    if not isinstance(env, RlEnv) and not isinstance(env, Wrapper):
        raise TypeError("'env' must be an instance of 'epyt_control.envs.RlEnv' or " +
                        f"'gymnasium.Wrapper' but not of '{type(env)}'")
    if isinstance(env, Wrapper):
        if not isinstance(env.env, RlEnv):
            raise TypeError("The wrapped environment must be an insance of " +
                            f"'epyt_control.envs.RlEnv' but not of '{type(env.env)}'")
    if not callable(policy):
        raise TypeError("'policy' must be callable -- " +
                        "i.e. mapping observations (numpy.ndarray) to actions (numpy.ndarray)")
    if not isinstance(n_max_iter, int) or n_max_iter < 1:
        raise ValueError("'n_max_iter' must be an integer >= 1")

    rewards = []
    scada_data = None

    obs, _ = env.reset()
    for _ in range(n_max_iter):
        action = policy(obs)
        obs, reward, terminated, _, info = env.step(action)
        if terminated is True:
            break

        rewards.append(reward)
        current_scada_data = info["scada_data"]
        if scada_data is None:
            scada_data = current_scada_data
        else:
            scada_data.concatenate(current_scada_data)

    env.close()

    return rewards, scada_data
