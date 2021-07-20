import types
from typing import Tuple, List, Callable, Union
import numpy as np

from ..utility import ActivationFunctions, ActivationFunctionsDerivatives
from savageml.utility import LossFunctions
from savageml.models import BaseModel


class MatrixNetModel(BaseModel):
    def __init__(self,
                 dimensions: List[int],
                 weight_range: Tuple[float, float] = (-2.0, 2.0),
                 activation_function: Callable = ActivationFunctions.SIGMOID,
                 activation_derivative: Callable = ActivationFunctionsDerivatives.SIGMOID_DERIVATIVE,
                 loss_function=LossFunctions.MSE,
                 weight_array: List[np.array] = None,
                 **kwargs):

        super().__init__(**kwargs)
        self.loss_function = loss_function
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

    def predict(self, x: np.ndarray, batch_size=None) -> np.ndarray:
        if batch_size is None:
            batch_size = x.shape[0]

        output = np.zeros((x.shape[0], self.dimensions[-1]))

        for index in range(0, x.shape[0], batch_size):
            batch = x[index: index + batch_size]

            for i in range(len(self.weight_array)):
                batch = np.concatenate([batch, np.ones((batch_size, 1))], axis=1)
                batch = self.activation_function(batch @ self.weight_array[i])

            output[index:index + batch_size, :] = batch
        return output

    def fit(self, x: Union[np.ndarray, types.GeneratorType, List], y: np.ndarray = None, batch_size=None):
        if y is not None:
            assert isinstance(x, np.ndarray), "If y is present, x must be a np array"
            assert y.shape[0] == x.shape[0], "x and y must have the same number of entries"
            self._fit_ndarray()
        else:


    def _fit_ndarray(self):
        pass

    def _fit_list(self):
        pass

    def _fit_generator(self):
        pass

    def _fit_batch(self, x: np.ndarray, y: np.ndarray):
        assert y.shape[0] == x.shape[0], "x and y must have the same number of entries"
        assert y.shape[1] >= self.dimensions[-1], "y entries too small"
        assert y.shape[1] <= self.dimensions[-1], "y entries too large"
        assert x.shape[1] >= self.dimensions[0], "x entries too small"
        assert x.shape[1] <= self.dimensions[0], "x entries too large"

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

    # def update(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, scale_dot: int = 5):
    #     self.screen = screen
    #     self.x = x
    #     self.y = y
    #     self.width = width
    #     self.height = height
    #     self.scale_dot = scale_dot
    #     self.scale_y = (self.height - self.scale_dot * 2) // max(self.dimensions)
    #     self.scale_x = (self.width - self.scale_dot * 2) // (len(self.dimensions) - 1)
    # 
    #     self.in_screen = [self.screen] * len(self.input_array)
    #     self.in_scale = [self.scale_dot] * len(self.input_array)
    # 
    #     self.in_loc = np.zeros((self.in_dem, 2)).astype(int)
    #     self.in_loc[:, 0:1] = np.add(self.in_loc[:, 0:1], self.x + self.scale_dot)
    #     self.in_loc[:, 1:2] = np.add(self.y + self.scale_dot, np.multiply(self.scale_y,
    #                                                                             np.add(self.in_loc[:, 1:2],
    #                                                                                       np.reshape(
    #                                                                                           range(self.in_dem),
    #                                                                                           (self.in_dem, 1)))))
    # 
    #     self.layers_color_formulas = [self.color_formula] * len(self.nodes_value_array)
    #     self.layers_screen = [[self.screen] * len(self.nodes_value_array[i])
    #                           for i in range(len(self.nodes_value_array))]
    #     self.layers_scale = [[self.scale_dot] * len(self.nodes_value_array[i]) for i in
    #                          range(len(self.nodes_value_array))]
    # 
    #     layers_loc_x = np.concatenate((
    #         np.add(self.x + self.scale_dot, np.multiply(self.scale_x,
    #                                                           np.reshape(range(1, len(self.nodes_value_array) + 1),
    #                                                                         (len(self.nodes_value_array), 1, 1)))),
    #         np.zeros((len(self.nodes_value_array), 1, 1))
    #     ), 2)
    # 
    #     self.layers_loc = [np.add(np.concatenate((
    #         np.zeros((len(self.nodes_value_array[x_]), 1)),
    #         np.add(self.y + self.scale_dot, np.multiply(self.scale_y,
    #                                                           np.reshape(range(len(self.nodes_value_array[x_])),
    #                                                                         (len(self.nodes_value_array[x_]),
    #                                                                          1))))), axis=1),
    #         layers_loc_x[x_]).astype(int) for x_ in range(len(self.nodes_value_array))]
    # 
    #     self.line_screen = [[[self.screen] * len(self.weight_array[x_][y_])
    #                          for y_ in range(len(self.nodes_value_array[x_]))]
    #                         for x_ in range(len(self.nodes_value_array))]
    #     self.line_scale = [[[1] * len(self.weight_array[x_][y_])
    #                         for y_ in range(len(self.nodes_value_array[x_]))]
    #                        for x_ in range(len(self.nodes_value_array))]
    # 
    #     self.line_color_formulas = [color_formula_line_helper] * len(self.weight_array)
    #     self.line_draw_formulas = [draw_line_helper] * len(self.weight_array)
    # 
    #     self.line_location_start = [[[[self.x + self.scale_dot + (x_ + 1) * self.scale_x,
    #                                    self.y + self.scale_dot + y_ * self.scale_y]] * len(self.weight_array[x_][y_])
    #                                  for y_ in range(len(self.nodes_value_array[x_]))]
    #                                 for x_ in range(len(self.nodes_value_array))]
    #     self.line_location_end = [[[[self.x + self.scale_dot + x_ * self.scale_x,
    #                                  self.y + self.scale_dot + y2 * self.scale_y] for y2 in
    #                                 range(len(self.weight_array[x_][y_]))] for y_ in
    #                                range(len(self.nodes_value_array[x_]))] for x_ in
    #                               range(len(self.nodes_value_array))]
    # 
    # def update_colors(self):
    #     self.in_colors = list(map(self.color_formula, self.input_array))
    #     self.layers_colors = list(map(map_helper, self.layers_color_formulas, self.nodes_value_array))
    #     self.line_colors = list(map(map_helper, self.line_color_formulas, self.weight_array))
    # 
    # def draw(self):
    #     self.update_colors()
    #     any(map(draw_circle, self.in_screen, self.in_colors, self.in_loc, self.in_scale))
    #     any(map(draw_circle_helper, self.layers_screen, self.layers_colors, self.layers_loc, self.layers_scale))
    #     any(map(map_helper_clean, self.line_draw_formulas, self.line_screen, self.line_colors, self.line_location_start,
    #             self.line_location_end, self.line_scale, self.line_scale))
