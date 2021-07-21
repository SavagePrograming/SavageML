from enum import Enum

import numpy as np


def mean_squared_error(observed_value: np.ndarray, predicted_value: np.ndarray, axis: tuple = None) -> np.ndarray:
    if axis is None:
        return np.mean(np.square(np.subtract(observed_value, predicted_value)))
    else:
        return np.mean(np.square(np.subtract(observed_value, predicted_value)), axis=axis, keepdims=True)


def mean_squared_error_derivative(observed_value: np.ndarray, predicted_value: np.ndarray, axis: tuple = None) -> np.ndarray:
    if axis is None:
        return np.multiply(np.mean(np.subtract(observed_value, predicted_value)), 2.0)
    else:
        return np.multiply(np.mean(np.subtract(observed_value, predicted_value), axis=axis, keepdims=True), 2.0)


class LossFunctions:
    MSE = mean_squared_error


class LossFunctionDerivatives:
    MSE_DERIVATIVE = mean_squared_error_derivative
