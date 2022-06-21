from typing import Tuple, List, Callable, Iterable, Union, Dict
import numpy as np

from savageml.utility import ActivationFunctions, ActivationFunctionsDerivatives
from savageml.utility import LossFunctions, LossFunctionDerivatives
from savageml.models import BaseModel
from savageml.utility import get_sample_from_iterator, batch_iterator, \
    batch_np_array


class LayerlessDenseNetModel(BaseModel):
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
    coordinates: Dict[int, Tuple[int, int]]
        Translates the name of a node to it's coordinates in the network
    weight_array: List[np.array], optional
        The values of the weights, if no value is supplied, randomly generated weights will be created.
    weight_array: List[np.array], optional
        A set of arrays representing where there are connections in the model.

    """

    coordinates: Dict[int, Tuple[int, int]]

    output_dimension: int
    hidden_dimension: int
    bias_dimension: int = 1
    input_dimension: int

    loss_function: Callable
    loss_function_derivative: Callable

    activation_function: Callable
    activation_derivative: Callable

    weight_range: Tuple[float, float]

    weight_array: List[np.ndarray]
    connections_array: List[np.ndarray]

    def __init__(self,
                 input_dimension: int,
                 hidden_dimension: int,
                 output_dimension: int,
                 weight_range: Tuple[float, float] = (-2.0, 2.0),
                 activation_function: Callable = ActivationFunctions.SIGMOID,
                 activation_derivative: Callable = ActivationFunctionsDerivatives.SIGMOID_DERIVATIVE,
                 loss_function=LossFunctions.MSE,
                 loss_function_derivative=LossFunctionDerivatives.MSE_DERIVATIVE,
                 weight_array: List[np.array] = None,
                 connections_array: List[np.array] = None,
                 coordinates: Dict[int, Tuple[int, int]] = None,
                 **kwargs):
        """Constructor Method"""

        super().__init__(**kwargs)

        self.coordinates = coordinates
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
        self.connections_array: List[np.array] = connections_array

        if self.coordinates is None:
            self.coordinates = {}
            for node in range(self.bias_dimension + self.input_dimension):
                coordinate_pair = (0, node)
                self.coordinates[node] = coordinate_pair
                self.coordinates[coordinate_pair] = node

            for node in range(self.hidden_dimension):
                coordinate_pair = (node + 1, 0)
                self.coordinates[node + self.bias_dimension + self.input_dimension] = coordinate_pair
                self.coordinates[coordinate_pair] = node + self.bias_dimension + self.input_dimension

            for node in range(self.output_dimension):
                coordinate_pair = (1 + self.hidden_dimension, node)
                self.coordinates[
                    node + self.bias_dimension + self.input_dimension + self.hidden_dimension] = coordinate_pair
                self.coordinates[
                    coordinate_pair] = node + self.bias_dimension + self.input_dimension + self.hidden_dimension

        if self.connections_array is None:
            self.connections_array = []
            # Make hidden Weights
            for i in range(self.hidden_dimension):
                shape = (self.bias_dimension + self.input_dimension + i, 1)
                connections_array = np.ones(shape)
                self.connections_array.append(connections_array)

            # Make output weights
            shape = (self.bias_dimension + self.input_dimension + self.hidden_dimension, self.output_dimension)
            connections_array = np.ones(shape)

            self.connections_array.append(connections_array)

        if self.weight_array is None:
            self.weight_array = []
            # Make hidden Weights
            for i in range(self.hidden_dimension):
                shape = (self.bias_dimension + self.input_dimension + i, 1)
                weight_array = np.random.random(shape) * (self.weight_range[1] - self.weight_range[0]) + \
                               self.weight_range[0]
                self.weight_array.append(weight_array * self.connections_array[i])

            # Make output weights
            shape = (self.bias_dimension + self.input_dimension + self.hidden_dimension, self.output_dimension)
            weight_array = np.random.random(shape) * (self.weight_range[1] -
                                                      self.weight_range[0]) + self.weight_range[0]
            self.weight_array.append(weight_array * self.connections_array[-1])

    @staticmethod
    def from_connections_list(input_dimension: int, hidden_dimension: int, output_dimension: int,
                              connection_list: List[Tuple[int, int, float]], **kwargs):
        """
        Creates a new :class:`LayerlessDenseNetModel` from a list of connection tuples.
        These tuples are in the shape, (start node, end node, weight)
        Nodes are in the order input nodes, bias nodes, hidden nodes, output nodes.


        Parameters
        ----------
        input_dimension - int
            The number of input nodes in the network
        hidden_dimension - int
            The number of hidden nodes in the network
        output_dimension - int
            The number of output nodes in the network
        connection_list - List[Tuple[int, int, float]]
            A list of connection tuples, which can be used to build another layerless network
        **kwargs
        kwargs
            Accepts all arguments that the LayerlessSparseNetModel

        Returns
        -------
        LayerlessDenseNetModel
            A new LayerlessDenseNetModel with the shape described in the connection list
        """

        connection_list: Tuple[int, int, float] = sorted(connection_list)

        coordinates: Dict[int, Tuple[int, int]] = {}
        Dependencies: Dict[int, int] = {}

        weight_array: List[np.ndarray] = []
        connections_array: List[np.ndarray] = []

        layers: List[List[int]] = [[]]
        nodes: List[int] = sorted(
            {in_node for in_node, _, _ in connection_list}.union({out_node for _, out_node, _ in connection_list}))

        for in_node, out_node, _ in connection_list:
            assert in_node < out_node, "connections must go from smaller to larger"

            if out_node in Dependencies:
                Dependencies[out_node].add(in_node)
            else:
                Dependencies[out_node] = {in_node}

        for node in nodes:
            layer = 0
            if node in Dependencies:
                layer = max([coordinates[parent][0] for parent in Dependencies[node]]) + 1
            while len(layers) <= layer:
                layers.append([])
            index = len(layers[layer])
            coordinates[node] = (layer, index)
            coordinates[(layer, index)] = node
            layers[layer].append(node)

        layer_starts: List[int] = [0]
        for layer in layers[:-1]:
            layer_starts.append(layer_starts[-1] + len(layer))

        for layer_start, layer in zip(layer_starts[1:], layers[1:]):
            shape = (layer_start, len(layer))
            connections_array.append(np.zeros(shape))
            weight_array.append(np.zeros(shape))

        for in_node, out_node, weight in connection_list:
            in_layer, in_index = coordinates[in_node]
            out_layer, out_index = coordinates[out_node]

            layer_weights = weight_array[out_layer - 1]
            layer_connections = connections_array[out_layer - 1]
            layer_weights[layer_starts[in_layer] + in_index, out_index] = weight
            layer_connections[layer_starts[in_layer] + in_index, out_index] = 1.0

        kwargs["input_dimension"] = input_dimension
        kwargs["hidden_dimension"] = hidden_dimension
        kwargs["output_dimension"] = output_dimension

        kwargs["weight_array"] = weight_array
        kwargs["connections_array"] = connections_array
        kwargs["coordinates"] = coordinates

        return LayerlessDenseNetModel(**kwargs)

    def get_connections_list(self) -> List[Tuple[int, int, float]]:
        """

        Returns
        -------

        """
        nodes: List[int] = sorted(filter(lambda x: not isinstance(x, tuple), self.coordinates.keys()))
        connection_list = []
        for in_node_idx in range(len(nodes) - self.output_dimension):
            for out_node_idx in range(max(in_node_idx + 1, self.bias_dimension + self.input_dimension), len(nodes)):
                in_node = nodes[in_node_idx]

                out_node = nodes[out_node_idx]
                out_layer, out_idx = self.coordinates[out_node]
                connections = self.connections_array[out_layer - 1]

                if connections[in_node_idx, out_idx] == 1.0:
                    weights = self.weight_array[out_layer - 1]
                    connection_list.append((in_node, out_node, weights[in_node_idx, out_idx]))
        return connection_list

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

        output = new_node

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

        prediction = new_node

        current_derivative = np.zeros_like(layer)

        current_derivative[:, -1 * self.output_dimension:] = self.loss_function_derivative(y, prediction, axis=1)

        weights_update = []

        for weights in reversed(self.weight_array):
            size = weights.shape[1]

            result = layer[:, -1 * size:]
            layer = layer[:, :-1 * size]

            result_derivative = current_derivative[:, -1 * size:]
            current_derivative = current_derivative[:, :-1 * size]

            dl_da = result_derivative * self.activation_derivative(result)

            node_update = dl_da @ weights.T
            weight_update = layer.T @ dl_da

            weights_update.append(weight_update * learning_rate)
            current_derivative = current_derivative + node_update

        new_weights = []
        for weight_update, weights, connections in zip(reversed(weights_update), self.weight_array, self.connections_array):
            new_weights.append((weights + weight_update) * connections)
        self.weight_array = new_weights

        return current_derivative[:, :-1 * self.bias_dimension]
