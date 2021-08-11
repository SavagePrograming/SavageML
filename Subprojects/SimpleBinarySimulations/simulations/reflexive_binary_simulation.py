from typing import Tuple, Union, Any

import numpy as np
from numpy import ndarray

from savageml.simulations import BaseSimulation, SimulationState


class ReflexiveBinarySimulation(BaseSimulation):
    def __init__(self, shape=(1,), max_steps=10, loss_function=lambda x: x, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.max_steps = max_steps
        self.step_count = 0

    def step(self, visualize=False) -> Tuple[Union[ndarray, Any], Union[ndarray, Any], ndarray]:
        if self.step_count >= self.max_steps:
            self.state = SimulationState.COMPLETE
        else:
            self.step_count += 1
            if self.step_count >= self.max_steps:
                self.state = SimulationState.COMPLETE
            else:
                self.state = SimulationState.RUNNING
            sample = self.random.choice([0.0, 1.], self.shape)
            return sample, sample.copy(), np.array(0.0)
