"""
This module contains Multi Layer Perceptrons (MLP), also known as feedforward artifical
neural networks, and deep neural networks (DNNs) for estimating the state transition function.
"""
import numpy as np
from epyt_flow.topology import NetworkTopology
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .surrogates import StateTransitionModel


class SimpleMlpStateTransitionModel(StateTransitionModel):
    """
    Multi-layer perceptron state transition model.
    Implemented in `scikit-learn <https://scikit-learn.org/stable/index.html>`_.

    Parameters
    ----------
    hidden_layers_size : `list[int]`, optional
        Dimensionality of the hidden layers.

        The default is [128].
    activation : `str`, optional
        Activation function for the hidden layers.

        The default is 'tanh'
    max_iter : `int`, optional
        Maximum number of training itertions.

        The default is 500.
    """
    def __init__(self, hidden_layer_sizes: list[int] = [128],
                 activation: str = "tanh", max_iter: int = 500, normalize: bool = True, **kwds):
        self._wdn_topology = None
        self._input_size = None
        self._state_size = None
        self._normalize = normalize

        if self._normalize is True:
            self._scaler = StandardScaler()
        else:
            self._scaler = None

        self._mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                 activation=activation, max_iter=max_iter)

        super().__init__(**kwds)

    def init(self, wdn_topology: NetworkTopology, input_size: int, state_size: int) -> None:
        self._wdn_topology = wdn_topology
        self._input_size = input_size
        self._state_size = state_size

    def fit(self, cur_state: np.ndarray, next_time_varying_quantity: np.ndarray,
            next_state: np.ndarray) -> None:
        X = np.concatenate((cur_state, next_time_varying_quantity), axis=1)

        if self._normalize is True:
            X = self._scaler.fit_transform(X)

        self._mlp.fit(X, next_state)

    def partial_fit(self, cur_state: np.ndarray, next_time_varying_quantity: np.ndarray,
                    next_state: np.ndarray) -> None:
        X = np.concatenate((cur_state, next_time_varying_quantity), axis=1)

        if self._normalize is True:
            self._scaler.partial_fit(X)
            X = self._scaler.transform(X)

        self._mlp.partial_fit(X, next_state)

    def predict(self, cur_state: np.ndarray,
                next_time_varying_quantity: np.ndarray) -> np.ndarray:
        X = np.concatenate((cur_state, next_time_varying_quantity), axis=1)

        if self._normalize is True:
            X = self._scaler.transform(X)

        return self._mlp.predict(X)
