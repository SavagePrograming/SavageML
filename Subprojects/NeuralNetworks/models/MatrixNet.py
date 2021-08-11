from typing import Tuple, List, Callable

import numpy, random

from Nets.Net import Net
from formulas import distance_formula, sigmoid, sigmoid_der, color_formula, draw_circle, map_helper, \
    draw_circle_helper, color_formula_line_helper, draw_line_helper, dim, map_helper_clean


class MatrixNet(Net):
    def __init__(self,
                 dimensions: List[int],
                 weight_range: Tuple[float, float],
                 activation: Callable = sigmoid,
                 activation_der: Callable = sigmoid_der,
                 color_formula_param: Callable = color_formula):

        super(MatrixNet, self).__init__(dimensions[0], dimensions[-1], activation, activation_der, color_formula_param)
        self.input_array: numpy.array = numpy.array([[0]] * dimensions[0])
        self.weight_array: List[numpy.array] = []
        self.nodes_value_array: List[List[List[int]]] = []
        self.dimensions: List[int] = dimensions
        self.score: float = 0

        for i in range(1, len(dimensions)):
            node_array = []
            weight_array = []

            for ii in range(dimensions[i]):
                node_array.append([0])
                weight_array.append([])
                for iii in range(dimensions[i - 1] + 1):
                    weight_array[ii].append(random.random() * (weight_range[1] - weight_range[0]) + weight_range[0])

            self.weight_array.append(numpy.array(weight_array))
            self.nodes_value_array.append(numpy.array(node_array))

    def set_in(self, array: List[int]):
        if len(array) == len(self.input_array):
            for i in range(len(array)):
                if array[i] is not None:
                    self.input_array[i][0] = array[i]

    def get_out(self):
        self.nodes_value_array[0] = self.activation_function(
            self.weight_array[0].dot(
                numpy.reshape(numpy.append(self.input_array, 1.0), ((len(self.input_array) + 1), 1))))

        for i in range(1, len(self.nodes_value_array)):
            self.nodes_value_array[i] = self.activation_function(self.weight_array[i].dot(
                numpy.reshape(numpy.append(self.nodes_value_array[i - 1], 1.0),
                              ((len(self.nodes_value_array[i - 1]) + 1), 1))))
        return self.nodes_value_array[-1]

    def learn(self, ratio: float, target: List[int]):
        target_length = len(target)

        target = numpy.reshape(numpy.array([target]), (target_length, 1))

        past = numpy.multiply(2.0, (numpy.subtract(target, self.nodes_value_array[-1])))

        error = distance_formula(target, self.nodes_value_array[-1])

        for i in range(len(self.nodes_value_array) - 1, 0, -1):
            nodes_value_array_temp = self.nodes_value_array[i]

            nodes_value_array_temp2 = numpy.reshape(numpy.append(self.nodes_value_array[i - 1], 1),
                                                    (1, len(self.nodes_value_array[i - 1]) + 1))

            sigmoid_derivative = self.activation_derivative(nodes_value_array_temp)
            sigmoid_derivative_with_past = numpy.multiply(sigmoid_derivative, past)
            current = sigmoid_derivative_with_past.dot(nodes_value_array_temp2)
            past = numpy.transpose(sigmoid_derivative_with_past).dot(self.weight_array[i])
            past = numpy.reshape(past, (len(past[0]), 1))[:-1]
            current = numpy.multiply(current, ratio)
            self.weight_array[i] = numpy.add(self.weight_array[i], current)

        nodes_value_array_temp = self.nodes_value_array[0]

        nodes_value_array_temp2 = numpy.reshape(numpy.append(self.input_array, 1),
                                                (1, len(self.input_array) + 1))
        sigmoid_derivative = self.activation_derivative(nodes_value_array_temp)
        sigmoid_derivative_with_past = numpy.multiply(sigmoid_derivative, past)

        current = sigmoid_derivative_with_past.dot(nodes_value_array_temp2)
        current = numpy.multiply(current, ratio)
        self.weight_array[0] = numpy.add(self.weight_array[0], current)

        return error

    def update(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, scale_dot: int = 5):
        self.screen = screen
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.scale_dot = scale_dot
        self.scale_y = (self.height - self.scale_dot * 2) // max(self.dimensions)
        self.scale_x = (self.width - self.scale_dot * 2) // (len(self.dimensions) - 1)

        self.in_screen = [self.screen] * len(self.input_array)
        self.in_scale = [self.scale_dot] * len(self.input_array)

        self.in_loc = numpy.zeros((self.in_dem, 2)).astype(int)
        self.in_loc[:, 0:1] = numpy.add(self.in_loc[:, 0:1], self.x + self.scale_dot)
        self.in_loc[:, 1:2] = numpy.add(self.y + self.scale_dot, numpy.multiply(self.scale_y,
                                                                                numpy.add(self.in_loc[:, 1:2],
                                                                                          numpy.reshape(
                                                                                              range(self.in_dem),
                                                                                              (self.in_dem, 1)))))

        self.layers_color_formulas = [self.color_formula] * len(self.nodes_value_array)
        self.layers_screen = [[self.screen] * len(self.nodes_value_array[i])
                              for i in range(len(self.nodes_value_array))]
        self.layers_scale = [[self.scale_dot] * len(self.nodes_value_array[i]) for i in
                             range(len(self.nodes_value_array))]

        layers_loc_x = numpy.concatenate((
            numpy.add(self.x + self.scale_dot, numpy.multiply(self.scale_x,
                                                              numpy.reshape(range(1, len(self.nodes_value_array) + 1),
                                                                            (len(self.nodes_value_array), 1, 1)))),
            numpy.zeros((len(self.nodes_value_array), 1, 1))
        ), 2)

        self.layers_loc = [numpy.add(numpy.concatenate((
            numpy.zeros((len(self.nodes_value_array[x_]), 1)),
            numpy.add(self.y + self.scale_dot, numpy.multiply(self.scale_y,
                                                              numpy.reshape(range(len(self.nodes_value_array[x_])),
                                                                            (len(self.nodes_value_array[x_]),
                                                                             1))))), axis=1),
            layers_loc_x[x_]).astype(int) for x_ in range(len(self.nodes_value_array))]

        self.line_screen = [[[self.screen] * len(self.weight_array[x_][y_])
                             for y_ in range(len(self.nodes_value_array[x_]))]
                            for x_ in range(len(self.nodes_value_array))]
        self.line_scale = [[[1] * len(self.weight_array[x_][y_])
                            for y_ in range(len(self.nodes_value_array[x_]))]
                           for x_ in range(len(self.nodes_value_array))]

        self.line_color_formulas = [color_formula_line_helper] * len(self.weight_array)
        self.line_draw_formulas = [draw_line_helper] * len(self.weight_array)

        self.line_location_start = [[[[self.x + self.scale_dot + (x_ + 1) * self.scale_x,
                                       self.y + self.scale_dot + y_ * self.scale_y]] * len(self.weight_array[x_][y_])
                                     for y_ in range(len(self.nodes_value_array[x_]))]
                                    for x_ in range(len(self.nodes_value_array))]
        self.line_location_end = [[[[self.x + self.scale_dot + x_ * self.scale_x,
                                     self.y + self.scale_dot + y2 * self.scale_y] for y2 in
                                    range(len(self.weight_array[x_][y_]))] for y_ in
                                   range(len(self.nodes_value_array[x_]))] for x_ in
                                  range(len(self.nodes_value_array))]

    def update_colors(self):
        self.in_colors = list(map(self.color_formula, self.input_array))
        self.layers_colors = list(map(map_helper, self.layers_color_formulas, self.nodes_value_array))
        self.line_colors = list(map(map_helper, self.line_color_formulas, self.weight_array))

    def draw(self):
        self.update_colors()
        any(map(draw_circle, self.in_screen, self.in_colors, self.in_loc, self.in_scale))
        any(map(draw_circle_helper, self.layers_screen, self.layers_colors, self.layers_loc, self.layers_scale))
        any(map(map_helper_clean, self.line_draw_formulas, self.line_screen, self.line_colors, self.line_location_start,
                self.line_location_end, self.line_scale, self.line_scale))

    def save(self) -> str:
        dim_save = ",".join(map(str, self.dimensions))
        weight_save = ",".join([";".join([":".join(map(str, node)) for node in row]) for row in self.weight_array])
        save_string = "%s|%s" % (dim_save, weight_save)
        return save_string

    def load(self, save):
        save = save.split("|")
        dim_save = save[0]
        dim_save = list(map(int, dim_save.split(",")))
        weight_save = save[1]
        weight_save = [numpy.array([list(map(float, node.split(":"))) for node in row.split(";")]) for row in
                       weight_save.split(",")]

        self.__init__(dim_save, (0.0, 0.0), activation=self.activation_function,
                      activation_der=self.activation_derivative,
                      color_formula_param=self.color_formula)
        self.weight_array = weight_save
