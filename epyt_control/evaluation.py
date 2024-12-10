"""
This module provides functions for evaluating policies/agents/control strategies on environments.
"""
from typing import Callable
import numpy as np
from epyt_flow.simulation import ScadaData

from .envs import RlEnv


def evaluate_policy(env: RlEnv, policy: Callable[[np.ndarray], np.ndarray],
                    n_max_iter: int = 1) -> tuple[list[float], ScadaData]:
    """
    Evaluates a given policy/agent/control strategy for a given environment --
    i.e. the policy/agent is applied to the environment and the rewards and ScadaData observations
    over time are recorded.

    Parameters
    ----------
    env : :class:`~epyt_control.envs.rl_env.RlEnv`
        The environment -- note that 'autoreset' must be set to False.
    policy : `Callable[[numpy.ndarray], numpy.ndarray]`
        Policy/Agent/Control strategy to be evaluated.
    n_max_iter : `int`, optional
        Upper bound on the number of iterations that is used for evaluating the given policy/agent.

        The default is 1.

    Returns
    -------
    `tuple[list[float], epyt_flow.simulation.ScadaData]`
        Tuple of rewards over time and a
        `epyt_flow.simulation.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_ 
        instance containing the WDN states over time.
    """
    if not isinstance(env, RlEnv):
        raise TypeError("'env' must be an instance of 'epyt_contro.envs.RlEnv' "+
                        f"but not of '{type(env)}'")
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

        rewards.append(reward)
        current_scada_data = info["scada_data"]
        if scada_data is None:
            scada_data = current_scada_data
        else:
            scada_data.concatenate(current_scada_data)

        if terminated:
            break

    env.close()

    return rewards, scada_data
