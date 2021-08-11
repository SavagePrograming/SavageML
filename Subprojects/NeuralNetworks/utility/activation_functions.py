from typing import Union

import numpy as np


def sigmoid_der(y: Union[int, float, np.array]) -> Union[int, float, np.array]:
    return np.multiply(np.subtract(1.0, y), y)


def tanh_derivative(y: Union[int, float, np.array]) -> Union[int, float, np.array]:
    return np.subtract(1.0, np.square(y))


def sigmoid(x: Union[int, float, np.array]) -> Union[int, float, np.array]:
    return np.divide(1.0, np.add(1.0, np.exp(np.negative(x))))


def relu(x: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(x, np.ndarray):
        y = x.copy()
        y[x < 0] = 0.0
        return y
    elif isinstance(x, float):
        return x if x > 0.0 else 0.0
    else:
        return x if x > 0 else 0


def relu_derivative(y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    if isinstance(y, np.ndarray):
        dx = y.copy()
        dx[y > 0] = 1.0
        return y
    elif isinstance(y, float):
        return 1.0 if y > 0.0 else 0.0
    else:
        return 1 if y > 0 else 0


class ActivationFunctions:
    TANH = np.tanh
    SIGMOID = sigmoid
    RELU = relu


class ActivationFunctionsDerivatives:
    TANH_DERIVATIVE = tanh_derivative
    SIGMOID_DERIVATIVE = sigmoid_der
    RELU_DERIVATIVE = relu_derivative

