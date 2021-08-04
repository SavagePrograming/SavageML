from typing import Tuple, List, Callable, Iterable, Union, Dict
import numpy as np

from savageml.utility import ActivationFunctions, ActivationFunctionsDerivatives
from savageml.utility import LossFunctions, LossFunctionDerivatives
from savageml.models import BaseModel
from savageml.utility import get_sample_from_iterator, batch_iterator, \
    batch_np_array


class LayerlessSparseNetModel(BaseModel):
    """A Layerless neural network, with sparsely packed hidden wights

    The layerless networks are meant to be able to represent various networks with non standard shapes.
    They can any network shape that is not cyclical.

    The sparse model has 4 sets of weights:
     * Input to Output
     * Input to Hidden
     * Hidden to Hidden, limited to above the diagonal, everything at or below the diagonal must be 0
     * Hidden to Output

    The equations for the layers are as follows:
     * :math:`H = \\sigma ([I \oplus 1 ] * W_{io} + H * W_{hh})` This needs to be repeated until stable
     * :math:`O = \\sigma ([I \oplus 1 ] * W_{io} + H * W_{ho})`

    +-------------------+------------------------------------+
    | Symbol            | Meaning                            |
    +===================+====================================+
    | :math:`W_{io}`    | Input to output weights            |
    +-------------------+------------------------------------+
    | :math:`W_{ih}`    | Input to hidden weights            |
    +-------------------+------------------------------------+
    | :math:`W_{hh}`    | Hidden to hidden weights           |
    +-------------------+------------------------------------+
    | :math:`W_{ho}`    | Hidden to output weights           |
    +-------------------+------------------------------------+
    | :math:`\\sigma`    | The activation function            |
    +-------------------+------------------------------------+
    | :math:`H`         | The hidden nodes for the network   |
    +-------------------+------------------------------------+
    | :math:`I`         | The input to the network           |
    +-------------------+------------------------------------+
    | :math:`O`         | The output of the network          |
    +-------------------+------------------------------------+


    Parameters
    ----------
    input_dimension
        The number of input nodes in the network
    hidden_dimension
        The number of hidden nodes in the network
    output_dimension
        The number of output nodes in the network
    weight_range: Tuple[float, float], optional
        The minimum and maximum values for randomly generated weight values
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
    input_output_weights - np.ndarray, optional
        The values of the input to output weights, if no value is supplied, randomly generated weights will be created.
    input_hidden_weights - np.ndarray, optional
        The values of the input to hidden weights, if no value is supplied, randomly generated weights will be created.
    hidden_hidden_weights - np.ndarray, optional
        The values of the hidden to hidden weights, if no value is supplied, randomly generated weights will be created.
    hidden_output_weights - np.ndarray, optional
        The values of the hidden to output weights, if no value is supplied, randomly generated weights will be created.
    """

    output_dimension: int
    hidden_dimension: int
    bias_dimension: int = 1
    input_dimension: int
    loss_function: Callable
    loss_function_derivative: Callable
    activation_function: Callable
    activation_derivative: Callable
    weight_range: Tuple[float, float]
    input_output_connections: np.ndarray
    input_output_weights: np.ndarray
    input_hidden_connections: np.ndarray
    input_hidden_weights: np.ndarray
    hidden_hidden_connections: np.ndarray
    hidden_hidden_weights: np.ndarray
    hidden_output_connections: np.ndarray
    hidden_output_weights: np.ndarray

    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: int,
                 output_dimension: int,
                 weight_range: Tuple[float, float] = (-2.0, 2.0),
                 activation_function: Callable = ActivationFunctions.SIGMOID,
                 activation_derivative: Callable = ActivationFunctionsDerivatives.SIGMOID_DERIVATIVE,
                 loss_function=LossFunctions.MSE,
                 loss_function_derivative=LossFunctionDerivatives.MSE_DERIVATIVE,
                 input_output_connections: np.array = None,
                 input_output_weights: np.array = None,
                 input_hidden_connections: np.array = None,
                 input_hidden_weights: np.array = None,
                 hidden_hidden_connections: np.array = None,
                 hidden_hidden_weights: np.array = None,
                 hidden_output_connections: np.array = None,
                 hidden_output_weights: np.array = None,
                 **kwargs):
        """Constructor Method"""
        super().__init__(**kwargs)

        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension
        self.bias_dimension = 1
        self.input_dimension = input_dimension

        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

        self.weight_range = weight_range

        self.input_output_connections: np.ndarray = input_output_connections
        self.input_output_weights: np.ndarray = input_output_weights

        self.input_hidden_connections: np.ndarray = input_hidden_connections
        self.input_hidden_weights: np.ndarray = input_hidden_weights

        self.hidden_hidden_connections: np.ndarray = hidden_hidden_connections
        self.hidden_hidden_weights: np.ndarray = hidden_hidden_weights

        self.hidden_output_connections: np.ndarray = hidden_output_connections
        self.hidden_output_weights: np.ndarray = hidden_output_weights

        # Working on input output
        if self.input_output_weights is None:
            shape = (self.bias_dimension + self.input_dimension, self.output_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            self.input_output_weights = weight_array
        if self.input_output_connections is None:
            self.input_output_connections = np.ones_like(self.input_output_weights)
        self.input_output_weights = self.input_output_weights * self.input_output_connections

        # Working on input hidden
        if self.input_hidden_weights is None:
            shape = (self.bias_dimension + self.input_dimension, self.hidden_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            self.input_hidden_weights = weight_array
        if self.input_hidden_connections is None:
            self.input_hidden_connections = np.ones_like(self.input_hidden_weights)
        self.input_hidden_weights = self.input_hidden_weights * self.input_hidden_connections

        # Working on hidden hidden
        if self.hidden_hidden_weights is None:
            shape = (self.hidden_dimension, self.hidden_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            self.hidden_hidden_weights = weight_array
        if self.hidden_hidden_connections is None:
            X, Y = np.meshgrid(range(self.hidden_dimension), range(self.hidden_dimension))
            self.hidden_hidden_connections = (X > Y) * 1.0
        self.hidden_hidden_weights = self.hidden_hidden_weights * self.hidden_hidden_connections

        # Working on hidden output
        if self.hidden_output_weights is None:
            shape = (self.hidden_dimension, self.output_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            self.hidden_output_weights = weight_array
        if self.hidden_output_connections is None:
            self.hidden_output_connections = np.ones_like(self.hidden_output_weights)
        self.hidden_output_weights = self.hidden_output_weights * self.hidden_output_connections

    @staticmethod
    def from_connections_list():
        pass

    @staticmethod
    def get_connections_list():
        pass

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
        input = np.concatenate([x, np.ones((size, 1))], axis=1)

        hidden = self.activation_function(input @ self.input_hidden_weights)
        for _ in range(self.hidden_dimension):
            hidden_: np.ndarray = hidden

            hidden: np.ndarray = self.activation_function(
                input @ self.input_hidden_weights + hidden @ self.hidden_hidden_weights)

            if (hidden_ == hidden).all():
                break

        output = self.activation_function(input @ self.input_output_weights + hidden @ self.hidden_output_weights)

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
        input = np.concatenate([x, np.ones((size, 1))], axis=1)

        hidden = self.activation_function(input @ self.input_hidden_weights)
        for _ in range(self.hidden_dimension):
            hidden_: np.ndarray = hidden

            hidden: np.ndarray = self.activation_function(
                input @ self.input_hidden_weights + hidden @ self.hidden_hidden_weights)

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

            hidden_derivatives: np.ndarray = ((output_derivatives @ self.hidden_output_weights.T +
                                              hidden_derivatives @ self.hidden_hidden_weights.T) *
                                              self.activation_derivative(hidden))

            if (hidden_derivatives_ == hidden_derivatives).all():
                break

        input_hidden_weights_update = (input.T @ hidden_derivatives) * learning_rate
        hidden_hidden_weights_update = (hidden.T @ hidden_derivatives) * learning_rate

        input_derivatives = output_derivatives @ self.hidden_output_weights.T + \
                            hidden_derivatives @ self.hidden_hidden_weights.T

        self.input_hidden_weights = self.input_hidden_connections * (self.input_hidden_weights +
                                                                     input_hidden_weights_update)
        self.input_output_weights = self.input_output_connections * (self.input_output_weights +
                                                                     input_output_weights_update)
        self.hidden_hidden_weights = self.hidden_hidden_connections * (self.hidden_hidden_weights +
                                                                       hidden_hidden_weights_update)
        self.hidden_output_weights = self.hidden_output_connections * (self.hidden_output_weights +
                                                                       hidden_output_weights_update)

        return input_derivatives[:, :-1]
