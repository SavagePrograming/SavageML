from typing import Tuple, List, Callable, Iterable, Union
import numpy as np

from Subprojects.NeuralNetworks.utility import ActivationFunctions, ActivationFunctionsDerivatives
from savageml.utility import LossFunctions, LossFunctionDerivatives
from savageml.models import BaseModel
from Subprojects.NeuralNetworks.utility.generic_functions import get_sample_from_iterator


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
        self.dimensions = dimensions
        self.weight_range = weight_range

        self.weight_array: List[np.array] = weight_array
        self.dimensions: List[int] = dimensions

        if self.weight_array is None:
            self.weight_array = []
            for i in range(1, len(dimensions)):
                weight_array = np.random.random((dimensions[i - 1] + 1, dimensions[i])) * (
                        self.weight_range[1] - self.weight_range[0]) + self.weight_range[1]
                self.weight_array.append(weight_array)

    def predict(self, x: Union[np.ndarray, Iterable], batch_size=None, iteration_limit=None) -> np.ndarray:
        if isinstance(x, np.ndarray):
            if batch_size is None:
                batch_size = x.shape[0]

            output = np.zeros((x.shape[0], self.dimensions[-1]))

            for index in range(0, x.shape[0], batch_size):
                batch = x[index: index + batch_size]

                processed_batch = self._predict_batch(batch)

                output[index:index + batch_size, :] = processed_batch
            return output
        else:
            assert isinstance(x, Iterable)
            data_iterator = iter(x)
            if iteration_limit is None:
                iteration_limit = sum(1 for _ in iter(x))
            if batch_size is None:
                batch_size = iteration_limit

            has_sample, sample = get_sample_from_iterator(data_iterator)
            x_sample = sample[0]

            output = np.zeros((iteration_limit, self.dimensions[-1]))
            output_index = 0

            assert has_sample, "Data source should have at least one sample"
            assert isinstance(x_sample, np.ndarray), "x should be np array"

            x_batch = np.zeros((batch_size, x_sample.shape[0]))
            count = 0

            while has_sample and (iteration_limit is None or count < iteration_limit):
                x_batch[(count % batch_size):(count % batch_size) + 1, :] = x_sample
                count += 1

                if (count % batch_size) == 0:
                    prediction = self._predict_batch(x_batch)

                    x_batch[:, :] = 0
                    output[output_index: output_index+batch_size, :] = prediction
                    output_index += batch_size

                has_sample, sample = get_sample_from_iterator(data_iterator)
                if has_sample:
                    x_sample = sample[0]

            if (count % batch_size) > 0:
                prediction = self._predict_batch(x_batch[:count % batch_size, :])

                output[output_index: output_index + count % batch_size, :] = prediction
                output_index += count % batch_size
            return output[:output_index, :]

    def _predict_batch(self, x: np.ndarray):
        batch = x
        batch_size = x.shape[0]

        for i in range(len(self.weight_array)):
            batch = np.concatenate([batch, np.ones((batch_size, 1))], axis=1)
            batch = self.activation_function(batch @ self.weight_array[i])

        return batch

    def fit(self, x: Iterable, y: np.ndarray = None, learning_rate=0.01, batch_size=None, iteration_limit=None):
        if y is not None:
            assert isinstance(x, np.ndarray), "If y is present, x must be a np array"
            assert y.shape[0] == x.shape[0], "x and y must have the same number of entries"

            if iteration_limit is None:
                iteration_limit = x.shape[0]
            iteration_limit = min(iteration_limit, x.shape[0])

            if batch_size is None:
                batch_size = x.shape[0]

            for batch_start in range(0, iteration_limit, batch_size):
                self._fit_batch(x[batch_start: batch_start + batch_size, :],
                                y[batch_start: batch_start + batch_size, :],
                                learning_rate)
        else:
            data_iterator = iter(x)

            has_sample, sample = get_sample_from_iterator(data_iterator)
            x_sample, y_sample = sample[0], sample[1]

            assert has_sample, "Data source should have at least one sample"
            assert isinstance(x_sample, np.ndarray), "x should be np array"
            assert isinstance(y_sample, np.ndarray), "y should be np array"

            x_batch = np.zeros((batch_size, x_sample.shape[1]))
            y_batch = np.zeros((batch_size, y_sample.shape[1]))
            count = 0

            while has_sample and (iteration_limit is None or count < iteration_limit):
                x_batch[(count % batch_size):(count % batch_size) + 1, :] = x_sample
                y_batch[(count % batch_size):(count % batch_size) + 1, :] = y_sample
                count += 1

                if (count % batch_size) >= batch_size:
                    self._fit_batch(x_batch, y_batch, learning_rate)

                    x_batch[:, :] = 0
                    y_batch[:, :] = 0

                has_sample, sample = get_sample_from_iterator(data_iterator)
                x_sample, y_sample = sample[0], sample[1]

            if (count % batch_size) > 0:
                self._fit_batch(x_batch[:count % batch_size, :],
                                y_batch[:count % batch_size, :],
                                learning_rate)

    def _fit_batch(self, x: np.ndarray, y: np.ndarray, learning_rate):
        assert y.shape[0] == x.shape[0], "x and y must have the same number of entries"
        assert y.shape[1] >= self.dimensions[-1], "y entries too small"
        assert y.shape[1] <= self.dimensions[-1], "y entries too large"
        assert x.shape[1] >= self.dimensions[0], "x entries too small"
        assert x.shape[1] <= self.dimensions[0], "x entries too large"
        size = y.shape[0]

        layer_values = []
        layer_values_bias = []

        # Forward Propagation
        layer_values.append(x)

        for i in range(len(self.weight_array)):
            biased_batch = np.concatenate([layer_values[-1], np.ones((size, 1))], axis=1)
            layer_values_bias.append(biased_batch)
            batch = self.activation_function(biased_batch @ self.weight_array[i])
            layer_values.append(batch)

        prediction = layer_values[-1]
        loss_derivative = self.loss_function_derivative(y, prediction, axis=1)
        current_derivative = loss_derivative

        weights_update = [np.zeros_like(weight_array) for weight_array in self.weight_array]

        for i in range(len(self.weight_array) - 1, -1, -1):
            dL_da = current_derivative * self.activation_derivative(layer_values[i + 1])
            node_update = dL_da @ self.weight_array[i].T
            weight_update = layer_values_bias[i].T @ dL_da
            assert weight_update.shape[1] == self.weight_array[i].shape[1]
            assert node_update.shape[1] == layer_values_bias[i].shape[1]

            weights_update[i] = weight_update * learning_rate
            current_derivative = np.sum(weight_update, axis=0, keepdims=True)[:, -1]

        return current_derivative

    def _learn(self, ratio: float, target: List[int]):
        target_length = len(target)

        target = np.reshape(np.array([target]), (target_length, 1))

        past = np.multiply(2.0, (np.subtract(target, self.nodes_value_array[-1])))

        error = self.loss_function(target, self.nodes_value_array[-1])

        for i in range(len(self.nodes_value_array) - 1, 0, -1):
            nodes_value_array_temp = self.nodes_value_array[i]

            nodes_value_array_temp2 = np.reshape(np.append(self.nodes_value_array[i - 1], 1),
                                                 (1, len(self.nodes_value_array[i - 1]) + 1))

            sigmoid_derivative = self.activation_derivative(nodes_value_array_temp)
            sigmoid_derivative_with_past = np.multiply(sigmoid_derivative, past)
            current = sigmoid_derivative_with_past.dot(nodes_value_array_temp2)
            past = np.transpose(sigmoid_derivative_with_past).dot(self.weight_array[i])
            past = np.reshape(past, (len(past[0]), 1))[:-1]
            current = np.multiply(current, ratio)
            self.weight_array[i] = np.add(self.weight_array[i], current)

        nodes_value_array_temp = self.nodes_value_array[0]

        nodes_value_array_temp2 = np.reshape(np.append(self.input_array, 1),
                                             (1, len(self.input_array) + 1))
        sigmoid_derivative = self.activation_derivative(nodes_value_array_temp)
        sigmoid_derivative_with_past = np.multiply(sigmoid_derivative, past)

        current = sigmoid_derivative_with_past.dot(nodes_value_array_temp2)
        current = np.multiply(current, ratio)
        self.weight_array[0] = np.add(self.weight_array[0], current)

        return error
