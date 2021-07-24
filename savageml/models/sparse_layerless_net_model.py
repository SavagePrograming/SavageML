from typing import Tuple, List, Callable, Iterable, Union, Dict
import numpy as np

from savageml.utility import ActivationFunctions, ActivationFunctionsDerivatives
from savageml.utility import LossFunctions, LossFunctionDerivatives
from savageml.models import BaseModel
from savageml.utility import get_sample_from_iterator, batch_iterator, \
    batch_np_array


class SparseLayerlessNetModel(BaseModel):
    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: int,
                 output_dimension: int,
                 network_connections: Dict[int, List[int]] = None,
                 weight_range: Tuple[float, float] = (-2.0, 2.0),
                 activation_function: Callable = ActivationFunctions.SIGMOID,
                 activation_derivative: Callable = ActivationFunctionsDerivatives.SIGMOID_DERIVATIVE,
                 loss_function=LossFunctions.MSE,
                 loss_function_derivative=LossFunctionDerivatives.MSE_DERIVATIVE,
                 input_output_weights: np.array = None,
                 input_hidden_weights: np.array = None,
                 hidden_hidden_weights: np.array = None,
                 hidden_output_weights: np.array = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.network_connections = network_connections

        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.bias_dimension = 1
        self.input_dimension = input_dimension

        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

        self.weight_range = weight_range

        # self.weight_array: List[np.array] = weight_array
        self.input_output_weights: np.ndarray = input_output_weights
        self.input_hidden_weights: np.ndarray = input_hidden_weights
        self.hidden_hidden_weights: np.ndarray = hidden_hidden_weights
        self.hidden_output_weights: np.ndarray = hidden_output_weights

        self.hidden_hidden_weights_clear = np.zeros((self.hidden_dimension, self.hidden_dimension))
        for i in range(self.hidden_dimension):
            for j in range(i+1, self.hidden_dimension):
                self.hidden_hidden_weights_clear[i, j] = 1.0

        if self.input_output_weights is None:
            # Make output weights
            shape = (self.bias_dimension + self.input_dimension, self.output_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            if self.network_connections is not None:
                connection_array = np.zeros(shape)
                for i in range(self.input_dimension + self.bias_dimension):
                    for connection in self.network_connections[i]:
                        if connection >= self.bias_dimension + self.input_dimension + self.hidden_dimension:
                            index = i - (self.bias_dimension + self.input_dimension + self.hidden_dimension)
                            connection_array[connection, index] = 1.0
                weight_array = weight_array * connection_array
            self.input_output_weights = weight_array

        if self.input_hidden_weights is None:
            # Make output weights
            shape = (self.bias_dimension + self.input_dimension, self.hidden_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            if self.network_connections is not None:
                connection_array = np.zeros(shape)
                for i in range(self.input_dimension + self.bias_dimension):
                    for connection in self.network_connections[i]:
                        if self.bias_dimension + self.input_dimension <= connection < self.bias_dimension + self.input_dimension + self.hidden_dimension:
                            index = i - (self.bias_dimension + self.input_dimension)
                            connection_array[connection, index] = 1.0
                weight_array = weight_array * connection_array
            self.input_hidden_weights = weight_array

        if self.hidden_hidden_weights is None:
            # Make output weights
            shape = (self.hidden_dimension, self.hidden_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            if self.network_connections is not None:
                connection_array = np.zeros(shape)
                for i in range(self.hidden_dimension):
                    for connection in self.network_connections[self.bias_dimension + self.input_dimension + i]:
                        if self.bias_dimension + self.input_dimension <= connection < self.bias_dimension + self.input_dimension + self.hidden_dimension:
                            index = i - (self.bias_dimension + self.input_dimension)
                            connection_array[connection, index] = 1.0
                weight_array = weight_array * connection_array
            self.hidden_hidden_weights = self.hidden_hidden_weights_clear * weight_array

        if self.hidden_output_weights is None:
            # Make output weights
            shape = (self.hidden_dimension, self.output_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            if self.network_connections is not None:
                connection_array = np.zeros(shape)
                for i in range(self.hidden_dimension):
                    for connection in self.network_connections[
                        self.bias_dimension + self.input_dimension + self.hidden_dimension + i]:
                        if self.bias_dimension + self.input_dimension + self.hidden_dimension <= connection:
                            index = i - (self.bias_dimension + self.input_dimension + self.hidden_dimension)
                            connection_array[connection, index] = 1.0
                weight_array = weight_array * connection_array
            self.hidden_output_weights = weight_array

    def predict(self, x: Union[np.ndarray, Iterable], batch_size=1, iteration_limit=None) -> np.ndarray:
        if isinstance(x, np.ndarray):

            output = np.zeros((0, self.output_dimension))

            if iteration_limit is not None and x.shape[0] > iteration_limit:
                x = x[:iteration_limit]

            for batch in batch_np_array(x, batch_size):
                prediction = self._predict_batch(batch)

                output = np.concatenate([output, prediction], axis=0)

            return output
        else:
            assert isinstance(x, Iterable)

            output = np.zeros((0, self.output_dimension))

            for sample_batch in batch_iterator(x, batch_size):
                x_batch_list = [sample[0] for sample in sample_batch]
                x_batch = np.concatenate(x_batch_list, axis=0)

                prediction = self._predict_batch(x_batch)

                output = np.concatenate([output, prediction], axis=0)

            return output

    def _predict_batch(self, x: np.ndarray):
        size = x.shape[0]
        input = np.concatenate([x, np.ones((size, 1))], axis=1)

        hidden = self.activation_function(input @ self.input_hidden_weights)
        for _ in range(self.hidden_dimension):
            hidden_: np.ndarray = hidden

            hidden: np.ndarray = self.activation_function(input @ self.input_hidden_weights + hidden @ self.hidden_hidden_weights)

            if (hidden_ == hidden).all():
                break

        output = self.activation_function(input @ self.input_output_weights + hidden @ self.hidden_output_weights)

        return output

    def fit(self, x: Iterable, y: np.ndarray = None, learning_rate=0.01, batch_size=1, iteration_limit=None):
        if y is not None:
            assert isinstance(x, np.ndarray), "If y is present, x must be a np array"
            assert y.shape[0] == x.shape[0], "x and y must have the same number of entries"

            if iteration_limit is not None and x.shape[0] > iteration_limit:
                x = x[:iteration_limit]
                y = y[:iteration_limit]

            for x_batch, y_batch in zip(batch_np_array(x, batch_size), batch_np_array(y, batch_size)):
                self._fit_batch(x_batch, y_batch, learning_rate)
        else:
            for sample_batch in batch_iterator(x, batch_size):
                x_batch_list = [sample[0] for sample in sample_batch]
                y_batch_list = [sample[1] for sample in sample_batch]

                x_batch = np.concatenate(x_batch_list, axis=0)
                y_batch = np.concatenate(y_batch_list, axis=0)

                self._fit_batch(x_batch, y_batch, learning_rate)

    def _fit_batch(self, x: np.ndarray, y: np.ndarray, learning_rate):
        assert y.shape[0] == x.shape[0], "x and y must have the same number of entries"
        assert y.shape[1] >= self.output_dimension, "y entries too small"
        assert y.shape[1] <= self.output_dimension, "y entries too large"
        assert x.shape[1] >= self.input_dimension, "x entries too small"
        assert x.shape[1] <= self.input_dimension, "x entries too large"

        # Forward Propagation
        size = x.shape[0]
        input = np.concatenate([x, np.ones((size, 1))], axis=1)

        hidden = self.activation_function(input @ self.input_hidden_weights)
        for _ in range(self.hidden_dimension):
            hidden_: np.ndarray = hidden

            hidden: np.ndarray = self.activation_function(input @ self.input_hidden_weights + hidden @ self.hidden_hidden_weights)

            if (hidden_ == hidden).all():
                break

        prediction = self.activation_function(input @ self.input_output_weights + hidden @ self.hidden_output_weights)

        # Back propagation
        input_derivatives: np.ndarray
        hidden_derivatives: np.ndarray
        output_derivatives: np.ndarray

        current_derivative = self.loss_function_derivative(y, prediction, axis=1)

        output_derivatives = current_derivative * self.activation_derivative(prediction)

        input_output_weights_update = (input.T @ output_derivatives) * learning_rate
        hidden_output_weights_update = (hidden.T @ output_derivatives) * learning_rate

        hidden_derivatives = output_derivatives @ self.hidden_output_weights.T * self.activation_derivative(hidden)
        for _ in range(self.hidden_dimension):
            hidden_derivatives_: np.ndarray = hidden

            hidden_derivatives: np.ndarray = (output_derivatives @ self.hidden_output_weights.T +
                                             hidden_derivatives @ self.hidden_hidden_weights.T) *\
                                             self.activation_derivative(hidden)

            if (hidden_derivatives_ == hidden_derivatives).all():
                break

        input_hidden_weights_update = (input.T @ hidden_derivatives) * learning_rate
        hidden_hidden_weights_update = (self.hidden_hidden_weights_clear * (hidden.T @ hidden_derivatives)) * learning_rate

        input_derivatives = output_derivatives @ self.hidden_output_weights.T +\
                            hidden_derivatives @ self.hidden_hidden_weights.T

        self.input_hidden_weights = self.input_hidden_weights + input_hidden_weights_update
        self.input_output_weights = self.input_output_weights + input_output_weights_update
        self.hidden_hidden_weights = self.hidden_hidden_weights + hidden_hidden_weights_update
        self.hidden_output_weights = self.hidden_output_weights + hidden_output_weights_update

        return input_derivatives[:, :-1]
