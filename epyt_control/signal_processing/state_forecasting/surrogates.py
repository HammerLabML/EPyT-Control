"""
This module contains different state transition surrogate models.
"""
from abc import abstractmethod
from typing import Callable
import numpy as np
from epyt_flow.topology import NetworkTopology
from epyt_flow.simulation import ScadaData

from ...envs import RlEnv


class StateTransitionModel():
    """
    Abstract base class of state transition models used in a surrogte -- i.e. a deep neural network
    approximating the state transition functions.
    """
    @abstractmethod
    def init(self, wdn_topology: NetworkTopology, input_size: int, state_size: int) -> None:
        """
        Initializes the model.

        Parameters
        ----------
        wdn_topology : `epyt_flow.topology.NetworkTopology <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.html#epyt_flow.topology.NetworkTopology>`_
            Information about the topology of the WDN.
        input_size : `int`
            Dimensionality of the input -- i.e. current state + time varying inputs that are
            relevant for the state transition (incl. control inputs).
        state_size : `int`
            Dimensionality of the state to be predicted.
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, cur_state: np.ndarray, next_time_varying_quantity: np.ndarray,
            next_state: np.ndarray) -> None:
        """
        Fits the neural network to given state transition data.

        Parameters
        ----------
        cur_state : numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current state of the system.
        next_time_varying_quantity : numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Time varying events (incl. control signals) that are relevant for evolving the state.
        next_state : numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Next state -- is to be predicted based on the other two arguments.
        """
        raise NotImplementedError()

    @abstractmethod
    def partial_fit(self, cur_state: np.ndarray, next_time_varying_quantity: np.ndarray,
                    next_state: np.ndarray) -> None:
        """
        Performs a partial fit of the state transition surrogate to given data.

        Parameters
        ----------
        cur_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current state of the system.
        next_time_varying_quantity : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Time varying events (incl. control signals) that are relevant for evolving the state.
        next_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Next state -- is to be predicted based on the other two arguments.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, cur_state: np.ndarray,
                next_time_varying_quantity: np.ndarray) -> np.ndarray:
        """
        Predicts the next state based on the current state and
        time varying events such as control signals.

        Parameters
        ----------
        cur_state : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current state.
        next_time_varying_quantity : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Time varying events (incl. control signals) that are relevant for evolving the state.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Next state.
        """
        raise NotImplementedError()


class StateTransitionSurrogate():
    """
    Base class of state transition surrogates.

    Parameters
    ----------
    wdn_topology : `epyt_flow.topology.NetworkTopology <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.html#epyt_flow.topology.NetworkTopology>`_
        Information about the topology of the WDN.
    n_actuators : `int`
        Number of actuators -- i.e. control inputs.
    """
    def __init__(self, wdn_topology: NetworkTopology, n_actuators: int):
        self._wdn_topology = wdn_topology
        self._n_actuators = n_actuators

    @abstractmethod
    def fit_to_scada(self, scada_data: ScadaData, control_actions: np.ndarray) -> None:
        """
        Fits the state transition surrogate to given `SCADA data <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_.

        Parameters
        ----------
        scada_data : `epyt_flow.simulation.scada_data.ScadaData <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.simulation.scada.html#epyt_flow.simulation.scada.scada_data.ScadaData>`_
            SCADA data.
        control_actions : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Control signals at every time step.
        """
        raise NotImplementedError()

    def fit_to_env(self, env: RlEnv, n_max_iter: int = None,
                   policy: Callable[[np.ndarray], np.ndarray] = None) -> None:
        """
        Fits the state transition surrogate to a given control environment.

        Parameters
        ----------
        env : :class:`~epyt_control.envs.rl_env.RlEnv`
            Control environment.
        n_max_iter : `int`
            Maximum numbe of iterations used for data collection.
            Note that data collection stops if the environment terminates.
        policy : `Callable[[numpy.ndarray], numpy.ndarray]`
            A policy for mapping observations to actions (i.e. control signals) -- will be applied at each time step.
            If None, random actions are sampled from the action space.

            The default is None.
        """
        # Run the environment and collect SCADA data
        scada_data = None
        control_actions = []

        obs, _ = env.reset()
        for _ in range(n_max_iter):
            action = policy(obs) if policy is not None else env.action_space.sample()
            control_actions.append(action)
            obs, _, terminated, _, info = env.step(action)
            if terminated is True:
                break

            current_scada_data = info["scada_data"]
            if scada_data is None:
                scada_data = current_scada_data
            else:
                scada_data.concatenate(current_scada_data)

        env.close()

        # Fit state transition surrogate model
        self.fit_to_scada(scada_data, np.array(control_actions))
