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


class HydraulicStateTransitionSurrogate(StateTransitionSurrogate):
    """
    Surrogate for the hydraulic state transition function.

    Paramaters
    ----------
    wdn_topology : `epyt_flow.topology.NetworkTopology <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.html#epyt_flow.topology.NetworkTopology>`_
        Information about the topology of the WDN.
    n_actuators : `int`
        Dimensionality of the control signal.
    state_transition_model : :class:`StateTransitionModel`
        State transition model which is used as an approximation of the true state transition function.
        Usually, a neural network is used.
    """
    def __init__(self, wdn_topology: NetworkTopology, n_actuators: int,
                 state_transition_model: StateTransitionModel):
        super().__init__(wdn_topology, n_actuators)

        state_size = wdn_topology.get_number_of_nodes() + wdn_topology.get_number_of_links()
        input_size = state_size + wdn_topology.get_number_of_nodes() + n_actuators
        state_transition_model.init(self._wdn_topology,
                                    input_size=input_size,
                                    state_size=state_size)
        self._state_transition_model = state_transition_model

    def fit_to_scada(self, scada_data: ScadaData, control_actions: np.ndarray = None) -> None:
        if not isinstance(scada_data, ScadaData):
            raise TypeError("'scada_data' must be an instance of " +
                            f"'epyt_flow.simulation.ScadaData' but not of '{type(scada_data)}'")
        if self._n_actuators > 0 and control_actions is None:
            raise ValueError("'control_actions' can not be None if 'n_actuators' > 0")

        X_pressure = scada_data.get_data_pressures()
        X_flows = scada_data.get_data_flows()
        X_demands = scada_data.get_data_demands()
        X_controls = control_actions
        n_time_steps = X_pressure.shape[0]

        cur_state = np.concatenate((X_pressure[:n_time_steps-1, :],
                                    X_flows[:n_time_steps-1, :]), axis=1)
        if X_controls is not None:
            next_time_varying_quantity = np.concatenate((X_demands[1:, :],
                                                         X_controls[:n_time_steps-1, :]), axis=1)
        else:
            next_time_varying_quantity = X_demands[1:, :]
        next_state = np.concatenate((X_pressure[1:, :], X_flows[1:, :]), axis=1)

        self._state_transition_model.fit(cur_state, next_time_varying_quantity, next_state)

    def __call__(self, cur_pressure: np.ndarray, cur_flow: np.ndarray,
                 next_demand: np.ndarray, control_actions: np.ndarray) -> np.ndarray:
        return self.predict(cur_pressure, cur_flow, next_demand, control_actions)

    def predict(self, cur_pressure: np.ndarray, cur_flow: np.ndarray,
                next_demand: np.ndarray, control_actions: np.ndarray) -> np.ndarray:
        """
        Predcts how the current state evolves for the next time step.

        cur_pressure : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current pressure at every node.
        cur_flow : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current flow rate at every link.
        next_demand : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Demand at every node for the next time step.
        control_actions : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Control signal at the current time step.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Next state.
        """
        X_cur_state = np.concatenate((cur_pressure, cur_flow), axis=1)
        X_control = np.concatenate((next_demand, control_actions), axis=1)

        return self._state_transition_model.predict(X_cur_state, X_control)


class WaterQualityStateTransitionSurrogate(StateTransitionSurrogate):
    """
    Surrogate for the quality (e.g. water age, chemical concentration) state transition function.

    Paramaters
    ----------
    wdn_topology : `epyt_flow.topology.NetworkTopology <https://epyt-flow.readthedocs.io/en/stable/epyt_flow.html#epyt_flow.topology.NetworkTopology>`_
        Information about the topology of the WDN.
    n_actuators : `int`
        Dimensionality of the control signal.
    state_transition_model : :class:`StateTransitionModel`
        State transition model which is used as an approximation of the true state transition function.
        Usually, a neural network is used.
    """
    def __init__(self, wdn_topology: NetworkTopology, n_actuators: int,
                 state_transition_model: StateTransitionModel):
        super().__init__(wdn_topology, n_actuators)

        state_size = wdn_topology.get_number_of_nodes() + wdn_topology.get_number_of_links()
        input_size = state_size + wdn_topology.get_number_of_links() + n_actuators
        state_transition_model.init(self._wdn_topology,
                                    input_size=input_size,
                                    state_size=state_size)
        self._state_transition_model = state_transition_model

    def fit_to_scada(self, scada_data: ScadaData, control_actions: np.ndarray = None) -> None:
        if not isinstance(scada_data, ScadaData):
            raise TypeError("'scada_data' must be an instance of " +
                            f"'epyt_flow.simulation.ScadaData' but not of '{type(scada_data)}'")
        if self._n_actuators > 0 and control_actions is None:
            raise ValueError("'control_actions' can not be None if 'n_actuators' > 0")

        X_flows = scada_data.get_data_flows()
        X_nodes_quality = scada_data.get_data_nodes_quality()
        X_links_quality = scada_data.get_data_links_quality()
        X_controls = control_actions
        n_time_steps = X_flows.shape[0]

        cur_state = np.concatenate((X_nodes_quality[:n_time_steps-1, :],
                                    X_links_quality[:n_time_steps-1, :]), axis=1)
        if X_controls is not None:
            next_time_varying_quantity = np.concatenate((X_flows[1:, :],
                                                        X_controls[:n_time_steps-1, :]), axis=1)
        else:
            next_time_varying_quantity = X_flows[1:, :]
        next_state = np.concatenate((X_nodes_quality[1:, :], X_links_quality[1:, :]), axis=1)

        self._state_transition_model.fit(cur_state, next_time_varying_quantity, next_state)

    def __call__(self, cur_node_quality: np.ndarray, cur_link_quality: np.ndarray,
                 next_flow: np.ndarray, control_actions: np.ndarray) -> np.ndarray:
        return self.predict(cur_node_quality, cur_link_quality,
                            next_flow, control_actions)

    def predict(self, cur_node_quality: np.ndarray, cur_link_quality: np.ndarray,
                next_flow: np.ndarray, control_actions: np.ndarray) -> np.ndarray:
        """
        Predicts the next state (i.e. quality everywhere) based on the current state of the system,
        the next flow, and control signals.

        Parameters
        ----------
        cur_flow : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current flow rate at each link.
        cur_node_quality : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current quality at every node.
        cur_link_quality : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current quality at every link.
        next_demand : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Demand at every node for the next time step.
        control_actions : `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Control signal at the current time step.

        Returns
        -------
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Next state.
        """
        X_cur_state = np.concatenate((cur_node_quality, cur_link_quality), axis=1)
        X_control = np.concatenate((next_flow, control_actions), axis=1)

        return self._state_transition_model.predict(X_cur_state, X_control)
