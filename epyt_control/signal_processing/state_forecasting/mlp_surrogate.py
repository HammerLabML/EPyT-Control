"""
This module contains Multi Layer Perceptrons (MLP), also known as feedforward artifical
neural networks, and deep neural networks (DNNs) for estimating the state transition function.
"""
import pickle
import numpy as np
from epyt_flow.topology import NetworkTopology
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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


class DnnStateTransitionModel(StateTransitionModel):
    """
    Neural network state transition model.
    Implemented in `PyTorch <https://pytorch.org/>`_.

    Parameters
    ----------
    hidden_layers_size : `list[int]`, optional
        Dimensionality of the hidden layers.

        The default is [128].
    activation : `str`, optional
        Activation function for the hidden layers.

        The default is 'tanh'
    last_layer_activation : `str`, optional
        Activation function of the last layer.
        If None, no acitvation function will be applied in the last layer.

        The default is None.
    max_iter : `int`, optional
        Maximum number of training itertions.

        The default is 200.
    device : `str`, optional
        Device used for the computation.

        The default is 'cpu'
    normalization_layer : `bool`, optional
        If True, the first layer is a normalization layer.

        The default is True.
    normalize_input_output : `bool`, optional
        If True, input is scaled and the target as well -- i.e. the scaled state is predicted.
        Can not be used in conjunction with 'normalization_layer'.

        The default is false.
    dropout : `float`, optional
        Specifies the dropout probability of in the input layer.
        If 0, no dropout will be used.

        The default is 0.
    batch_size : `int`, optional
        Batch size for training. Be aware that the batch size might have an influence
        on the normalization layer.

        The default is 128
    """
    def __init__(self, hidden_layers_size: list[int] = [128],
                 activation: str = "tanh", last_layer_activation: str = None,
                 max_iter: int = 200, device: str = "cpu", normalization_layer: bool = True,
                 normalize_input_output: bool = False,
                 dropout: float = 0., batch_size: int = 128,
                 **kwds):
        self._hidden_layers_size = hidden_layers_size
        self._activation = activation
        self._last_layer_activation = last_layer_activation
        self._max_iter = max_iter
        self._device = device
        self._normalization_layer = normalization_layer
        self._normalize_input_output = normalize_input_output
        self._dropout = dropout
        self._batch_size = batch_size
        self._model = None
        self._wdn_topology = None
        self._input_size = None
        self._state_size = None
        self._scaler = None

        if normalization_layer is True and normalization_layer is True:
            raise ValueError("'normalization_layer' and 'normalization_layer' " +
                             "can not be used at the same time.")

        if normalize_input_output is True:
            self._scaler = StandardScaler()

        super().__init__(**kwds)

    def _get_activation_func(self, activation_desc: str) -> nn.Module:
        if activation_desc == "relu":
            return nn.ReLU()
        elif activation_desc == "tanh":
            return nn.Tanh()
        else:
            return None

    def init(self, wdn_topology: NetworkTopology, input_size: int, state_size: int) -> None:
        self._wdn_topology = wdn_topology
        self._input_size = input_size
        self._state_size = state_size

        layers = []

        if self._normalization_layer is True:
            layers.append(nn.BatchNorm1d(self._input_size,))

        if self._dropout > 0.:
            layers.append(nn.Dropout(p=self._dropout))

        layers.append(nn.Linear(self._input_size, self._hidden_layers_size[0]))
        for i in range(1, len(self._hidden_layers_size)):
            layers.append(self._get_activation_func(self._activation))
            layers.append(nn.Linear(self._hidden_layers_size[i-1], self._hidden_layers_size[i]))

        layers.append(self._get_activation_func(self._activation))
        layers.append(nn.Linear(self._hidden_layers_size[-1], self._state_size))

        if self._last_layer_activation is not None:
            layers.append(self._get_activation_func(self._last_layer_activation))

        self._model = nn.Sequential(*layers)

    def load_from_file(self, file_path: str) -> None:
        """
        Loads model's weights and the standard scaler from a given file.

        Parameters
        ----------
        file_path : `str`
            File path.
        """
        self._model = torch.load(file_path, weights_only=False)

        if self._normalize_input_output is True:
            with open(f"{file_path}.pickle", "rb") as f_in:
                self._scaler = pickle.load(f_in)

    def save_to_file(self, file_path: str) -> None:
        """
        Saves model's weights and the standard scaler to a file.

        Parameters
        ----------
        file_path : `str`
            File path.
        """
        torch.save(self._model, file_path)

        if self._normalize_input_output is True:
            with open(f"{file_path}.pickle", "wb") as f_in:
                pickle.dump(self._scaler, f_in)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def compute_jacobian(self, cur_state: np.ndarray,
                         next_time_varying_quantity: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian w.r.t. a given state (incl. control signals).

        Parameters
        ----------
        cur_state : numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Current state of the system.
        next_time_varying_quantity : numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Time varying events (incl. control signals) that are relevant for evolving the state.

        Returns
        -------
        numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Jacobian.
        """
        X = np.concatenate((cur_state, next_time_varying_quantity), axis=1)

        if self._normalize_input_output:
            X = self._scaler.transform(X)

        jac = torch.autograd.functional.jacobian(self._forward, torch.Tensor(X)).detach().cpu().numpy()

        return jac

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Wrap data
        X_train = torch.Tensor(X)
        y_train = torch.Tensor(y)

        train_data_set = TensorDataset(X_train, y_train)
        train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=self._batch_size)

        # Loss function and optimizer
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters())

        # Run training
        self._model.train()
        for _ in range(self._max_iter):
            for batch, (X, y) in enumerate(train_data_loader):
                X, y = X.to(self._device), y.to(self._device)

                # Compute prediction error
                pred = self._forward(X)
                loss = loss_func(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}")

        self._model.train(False)

    def fit(self, cur_state: np.ndarray, next_time_varying_quantity: np.ndarray,
            next_state: np.ndarray) -> None:
        X = np.concatenate((cur_state, next_time_varying_quantity), axis=1)

        if self._normalize_input_output:
            X = self._scaler.fit_transform(X)

            dummy_next_flows = np.zeros((next_state.shape[0], X.shape[1] - next_state.shape[1]))
            next_state_ = self._scaler.transform(np.concatenate((next_state, dummy_next_flows),
                                                                axis=1))
            next_state = next_state_[:, :next_state.shape[1]]

        self._fit(X, next_state)

    def partial_fit(self, cur_state: np.ndarray, next_time_varying_quantity: np.ndarray,
                    next_state: np.ndarray) -> None:
        raise NotImplementedError()

    def predict(self, cur_state: np.ndarray,
                next_time_varying_quantity: np.ndarray,
                invert_output_scaling: bool = False) -> np.ndarray:
        X = np.concatenate((cur_state, next_time_varying_quantity), axis=1)
        if self._normalize_input_output:
            X = self._scaler.transform(X)

        Y_pred = self._forward(torch.Tensor(X)).detach().cpu().numpy()

        if invert_output_scaling is True:
            return self.invert_output_scaling(Y_pred)
        else:
            return Y_pred

    def invert_output_scaling(self, Y_pred: np.ndarray) -> np.ndarray:
        """
        Inverts the scaling of the output (i.e. predicted state).

        Parameters
        ----------
        Y_pred : numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Predicted state.

        Returns
        -------
        numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
            Unscaled predicted state.
        """
        if self._normalize_input_output is not True:
            raise ValueError("Output is not scaled!")

        dummy_control = np.ones((Y_pred.shape[0], self._scaler.n_features_in_ - Y_pred.shape[1]))
        Y_pred_ = self._scaler.inverse_transform(np.concatenate((Y_pred, dummy_control), axis=1))
        return Y_pred_[:, :Y_pred.shape[1]]
