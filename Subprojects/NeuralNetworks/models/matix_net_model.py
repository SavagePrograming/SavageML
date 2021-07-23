from typing import Tuple, List, Callable, Iterable, Union
import numpy as np

from Subprojects.NeuralNetworks.utility import ActivationFunctions, ActivationFunctionsDerivatives
from savageml.utility import LossFunctions, LossFunctionDerivatives
from savageml.models import BaseModel
from Subprojects.NeuralNetworks.utility import get_sample_from_iterator, batch_iterator, \
    batch_np_array


class MatrixNetModel(BaseModel):
    def __init__(self,
                 dimensions: List[int],
                 weight_range: Tuple[float, float] = (-2.0, 2.0),
                 activation_function: Callable = ActivationFunctions.SIGMOID,
                 activation_derivative: Callable = ActivationFunctionsDerivatives.SIGMOID_DERIVATIVE,
                 loss_function=LossFunctions.MSE,
                 loss_function_derivative=LossFunctionDerivatives.MSE_DERIVATIVE,
                 weight_array: List[np.array] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.weight_range = weight_range

        self.weight_array: List[np.array] = weight_array
        self.dimensions: List[int] = dimensions

        if self.weight_array is None:
            self.weight_array = []
            for i in range(1, len(dimensions)):
                weight_array = np.random.random((dimensions[i - 1] + 1, dimensions[i])) * (
                        self.weight_range[1] - self.weight_range[0]) + self.weight_range[0]
                self.weight_array.append(weight_array)

    def predict(self, x: Union[np.ndarray, Iterable], batch_size=1, iteration_limit=None) -> np.ndarray:
        if isinstance(x, np.ndarray):

            output = np.zeros((0, self.dimensions[-1]))

            if iteration_limit is not None and x.shape[0] > iteration_limit:
                x = x[:iteration_limit]

            for batch in batch_np_array(x, batch_size):

                prediction = self._predict_batch(batch)

                output = np.concatenate([output, prediction], axis=0)

            return output
        else:
            assert isinstance(x, Iterable)

            output = np.zeros((0, self.dimensions[-1]))

            for sample_batch in batch_iterator(x, batch_size):
                x_batch_list = [sample[0] for sample in sample_batch]
                x_batch = np.concatenate(x_batch_list, axis=0)

                prediction = self._predict_batch(x_batch)

                output = np.concatenate([output, prediction], axis=0)

            return output

    def _predict_batch(self, x: np.ndarray):
        size = x.shape[0]
        layer = x

        for weights in self.weight_array:
            layer_bias = np.concatenate([layer, np.ones((size, 1))], axis=1)
            layer = self.activation_function(layer_bias @ weights)

        return layer

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
        assert y.shape[1] >= self.dimensions[-1], "y entries too small"
        assert y.shape[1] <= self.dimensions[-1], "y entries too large"
        assert x.shape[1] >= self.dimensions[0], "x entries too small"
        assert x.shape[1] <= self.dimensions[0], "x entries too large"
        layer_values = []
        layer_values_bias = []

        # Forward Propagation

        size = x.shape[0]
        layer = x

        for weights in self.weight_array:
            layer_bias = np.concatenate([layer, np.ones((size, 1))], axis=1)
            layer = self.activation_function(layer_bias @ weights)

            layer_values_bias.append(layer_bias)
            layer_values.append(layer)

        prediction = layer_values[-1]
        current_derivative = self.loss_function_derivative(y, prediction, axis=1)

        weights_update = []

        for result, layer, weights in zip(reversed(layer_values),
                                 reversed(layer_values_bias),
                                 reversed(self.weight_array)):
            dl_da = current_derivative * self.activation_derivative(result)
            node_update = dl_da @ weights.T
            weight_update = layer.T @ dl_da

            weights_update.append(weight_update * learning_rate)
            current_derivative = np.sum(node_update, axis=0, keepdims=True)[:, :-1]

        new_weights = []
        for weight_update, weights in zip(reversed(weights_update), self.weight_array):
            new_weights.append(weights + weight_update)

        self.weight_array = new_weights

        return current_derivative