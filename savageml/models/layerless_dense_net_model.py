from typing import Tuple, List, Callable, Iterable, Union, Dict
import numpy as np

from savageml.utility import ActivationFunctions, ActivationFunctionsDerivatives
from savageml.utility import LossFunctions, LossFunctionDerivatives
from savageml.models import BaseModel
from savageml.utility import get_sample_from_iterator, batch_iterator, \
    batch_np_array


class LayerlessDenseNetModel(BaseModel):
    """

    :param input_dimension:
    :param hidden_dimension:
    :param output_dimension:
    :param network_connections:
    :param weight_range:
    :param activation_function:
    :param activation_derivative:
    :param loss_function:
    :param loss_function_derivative:
    :param weight_array:
    :param kwargs:
    """

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
                 weight_array: List[np.array] = None,
                 **kwargs):
        """Constructor Method"""

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

        self.weight_array: List[np.array] = weight_array
        self.output_weights = None

        if self.weight_array is None:
            self.weight_array = []
            # Make hidden Weights
            for i in range(self.hidden_dimension):
                shape = (self.bias_dimension + self.input_dimension + i, 1)
                weight_array = np.random.random(shape) * (self.weight_range[1] - self.weight_range[0]) + \
                               self.weight_range[0]
                if self.network_connections is not None:
                    connection_array = np.zeros(shape)
                    for connection in self.network_connections[self.bias_dimension + self.input_dimension + i]:
                        connection_array[connection, 0] = 1.0
                    weight_array = weight_array * connection_array
                self.weight_array.append(weight_array)

            # Make output weights
            shape = (self.bias_dimension + self.input_dimension + self.hidden_dimension, self.output_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            if self.network_connections is not None:
                connection_array = np.zeros(shape)
                for i in range(self.output_dimension):
                    for connection in self.network_connections[self.bias_dimension +
                                                               self.input_dimension +
                                                               self.hidden_dimension + i]:
                        connection_array[connection, i] = 1.0
                weight_array = weight_array * connection_array
            self.weight_array.append(weight_array)
        self.output_weights = self.weight_array.pop(-1)

    def predict(self, x: Union[np.ndarray, Iterable], batch_size: int = 1, iteration_limit: int = None) -> np.ndarray:
        """Predicting values of some function

        Uses forward propagation to produce predicted values.

        Parameters
        ----------
        x - Union[np.ndarray, Iterable]
            The input values to the model, or an iterable that produces (x, ...) tuples.
        batch_size - int, optional
            The size of the batch of input to be processed at the same time. Defaults to 1
        iteration_limit - int, optional
            The maximum number of iterations to process.
            Defaults to None, which means there is no limit

        Returns
        -------
        np.ndarray
            The predicted values
        """
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
        layer = np.concatenate([x, np.ones((size, 1))], axis=1)

        for weights in self.weight_array:
            new_node = self.activation_function(layer @ weights)
            layer = np.concatenate([layer, new_node], axis=1)
        output = self.activation_function(layer @ self.output_weights)

        return output

    def fit(self, x: Union[np.ndarray, Iterable], y: np.ndarray = None, learning_rate: float = 0.01,
            batch_size: int = 1, iteration_limit: int = None):
        """ The function to fit the model to some data

        Uses forward propagation to estimate y values.
        Compares those values to the true values using the loss function,
        producing a gradient with the derivative of that function.
        Backpropagation is then used to update the networks weights.

        Parameters
        ----------
        x - Union[np.ndarray, Iterable]
            The input values to the model, or an iterable that produces (x, y, ...) tuples.
        y - np.ndarray, optional
            The output values to the model, not expected to be present if x is an Iterable.
        learning_rate - float, optional
            The rate at which the weights are updated, defaults to 0.01
        batch_size - int, optional
            The number of samples to be processed at once, defaults to 1
        iteration_limit - int, optional
            The maximum number of samples to be processed, defaults to no limit
        """
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
        layer = np.concatenate([x, np.ones((size, 1))], axis=1)

        for weights in self.weight_array:
            new_node = self.activation_function(layer @ weights)
            layer = np.concatenate([layer, new_node], axis=1)

        prediction = self.activation_function(layer @ self.output_weights)

        current_derivative = self.loss_function_derivative(y, prediction, axis=1)

        dl_da = current_derivative * self.activation_derivative(prediction)
        output_weights_update = (layer.T @ dl_da) * learning_rate
        current_derivative = dl_da @ self.output_weights.T

        weights_update = []

        for weights in reversed(self.weight_array):
            result = layer[:, -1:]
            layer = layer[:, :-1]

            result_derivative = current_derivative[:, -1:]
            current_derivative = current_derivative[:, :-1]

            dl_da = result_derivative * self.activation_derivative(result)

            node_update = dl_da @ weights.T
            weight_update = layer.T @ dl_da

            weights_update.append(weight_update * learning_rate)
            current_derivative = current_derivative + node_update

        new_weights = []
        for weight_update, weights in zip(reversed(weights_update), self.weight_array):
            new_weights.append(weights + weight_update)
        self.weight_array = new_weights

        self.output_weights = self.output_weights + output_weights_update

        return current_derivative[:, :-1]
