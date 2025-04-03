"""
This module contains different state transition surrogate models.
"""
from abc import abstractmethod
import numpy as np
from epyt_flow.topology import NetworkTopology


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
