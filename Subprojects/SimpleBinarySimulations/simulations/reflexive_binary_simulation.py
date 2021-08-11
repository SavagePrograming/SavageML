from typing import Tuple, Union, Any

import numpy as np
from numpy import ndarray

from savageml.simulations import BaseSimulation


class ReflexiveBinarySimulation(BaseSimulation):
    def __init__(self, shape=(1,), max_steps=10):
        super().__init__()
        self.shape = shape
        self.max_steps = max_steps
        self.step_count = 0

    def step(self, visualize=False) -> Tuple[Union[ndarray, Any], Union[ndarray, Any], ndarray]:
        if self.step_count >= self.max_steps:
            raise StopIteration
        else:
            self.step_count += 1
            sample = self.random.choice([0.0, 1.], self.shape)
            return (sample, sample.copy(), np.array(0.0))

    def __iter__(self):
        pass

    def reset(self):
        pass