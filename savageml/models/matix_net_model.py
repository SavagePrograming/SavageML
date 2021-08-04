from typing import Tuple, List, Callable, Iterable, Union
import numpy as np

from savageml.utility import ActivationFunctions, ActivationFunctionsDerivatives
from savageml.utility import LossFunctions, LossFunctionDerivatives
from savageml.models import BaseModel
from savageml.utility import get_sample_from_iterator, batch_iterator, \
    batch_np_array


class MatrixNetModel(BaseModel):
    """ A Simple Multilayer Perceptron

    The matrix net is a multilayer perceptron.
    It takes dimensions for each layer.
    It supports any number of layers, which can be any integer size greater than 0.
    The equations for the network can be seen here:

    :math:`X_{n+1} = \\sigma ([X_n \oplus 1 ] * W_n)`

    +-------------------+------------------------------------+
    | Symbol            | Meaning                            |
    +===================+====================================+
    | :math:`W_n`       | The weights for the layer          |
    +-------------------+------------------------------------+
    | :math:`\\sigma`    | The activation function            |
    +-------------------+------------------------------------+
    | :math:`X_n`       | The input for the layer            |
    +-------------------+------------------------------------+
    | :math:`X_0`       | The input to the network           |
    +-------------------+------------------------------------+
    | :math:`X_{final}` | The output of the network          |
    +-------------------+------------------------------------+


    Parameters
    ----------
    dimensions: List[int]
        The dimensions of each layer in the network
    weight_range: Tuple[float, float], optional
        The minimum and maximum values for randomly generated weight values
    weight_array: List[np.array], optional
        The values of the weights, if no value is supplied, randomly generated weights will be created.
    activation_function: Callable, optional
        The activation function for the network. Defaults to sigmoid.
        Remember to also set the activation derivative if you want the model to learn
    activation_derivative: Callable, optional
        The derivative of the activation function for the network.
        This is used in backpropagation.
        Defaults to derivative of a sigmoid.
        Remember to also set the activation function if you want the model to learn
    loss_function: Callable, optional
        The loss function of network, used to compare predictions to expected values.
        Defaults to Mean Squared Error.
        Remember to also set the loss derivative, or the network will not learn properly.
    loss_function_derivative: Callable, optional
        The derivative of the loss function of network, used in backpropagation.
        Defaults to the derivative of mean squared error.
    """

    loss_function: Callable
    loss_function_derivative: Callable

    activation_function: Callable
    activation_derivative: Callable

    weight_range: Tuple[float, float]
    weight_array: List[np.array]

    dimensions: List[int]

    def __init__(self,
                 dimensions: List[int],
                 weight_range: Tuple[float, float] = (-2.0, 2.0),
                 activation_function: Callable = ActivationFunctions.SIGMOID,
                 activation_derivative: Callable = ActivationFunctionsDerivatives.SIGMOID_DERIVATIVE,
                 loss_function=LossFunctions.MSE,
                 loss_function_derivative=LossFunctionDerivatives.MSE_DERIVATIVE,
                 weight_array: List[np.array] = None,
                 **kwargs):
        """Constructor Method"""

        super().__init__(**kwargs)
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.weight_range = weight_range

        self.weight_array = weight_array
        self.dimensions = dimensions

        if self.weight_array is None:
            self.weight_array = []
            for i in range(1, len(dimensions)):
                weight_array = np.random.random((dimensions[i - 1] + 1, dimensions[i])) * (
                        self.weight_range[1] - self.weight_range[0]) + self.weight_range[0]
                self.weight_array.append(weight_array)

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

    def fit(self, x: Union[np.ndarray, Iterable], y: np.ndarray = None, learning_rate=0.01, batch_size=1, iteration_limit=None):
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
