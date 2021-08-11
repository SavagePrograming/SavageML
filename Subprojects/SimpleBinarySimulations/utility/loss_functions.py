from enum import Enum

import numpy as np


def mean_squared_error(first_value: np.ndarray, second_value: np.ndarray, axis: tuple = None) -> np.ndarray:
    if axis is None:
        return np.sum(np.square(np.subtract(first_value, second_value)))
    else:
        return np.sum(np.square(np.subtract(first_value, second_value)), axis=axis)


class LossFunctions:
    MSE = mean_squared_error
