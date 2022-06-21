import numpy

from ..utility import LossFunctions
import numpy as np
from savageml.simulations import BaseSimulation, SimulationState


class BinaryAndSumSimulation(BaseSimulation):
    def __init__(self, shape=(1,), max_steps=10, loss_function=LossFunctions.MSE, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.max_steps = max_steps
        self.step_count = 0
        self.loss_function = loss_function
        self.last_out = numpy.ones((1,) + self.shape)

    def step(self, visualize=False) -> tuple:
        if self.step_count >= self.max_steps:
            self.state = SimulationState.COMPLETE
            return ()
        else:
            self.step_count += 1
            if self.step_count >= self.max_steps:
                self.state = SimulationState.COMPLETE
            else:
                self.state = SimulationState.RUNNING
            sample = self.random.choice([0.0, 1.], (1,) + self.shape)
            if visualize:
                print("sample", sample)
            self.last_out = self.last_out * sample
            if self.model is not None:
                prediction = self.model.predict(sample)
                loss = self.loss_function(self.last_out, prediction)
                if visualize:
                    print("prediction:", (prediction > 0.5) * 1.0)
                    print("expected:", self.last_out)
                    print("loss:", loss)
            else:
                loss = None
            return sample, self.last_out, loss
