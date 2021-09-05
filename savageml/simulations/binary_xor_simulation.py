from ..utility import LossFunctions
import numpy as np
from savageml.simulations import BaseSimulation, SimulationState


class BinaryXorSimulation(BaseSimulation):
    def __init__(self, shape=(2,), max_steps=10, loss_function=LossFunctions.MSE, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.max_steps = max_steps
        self.step_count = 0
        self.loss_function = loss_function

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
            output = np.array(((sample == 1.0).sum() == 1.0)).astype(float).reshape((1, 1))
            if self.model is not None:
                prediction = self.model.predict(sample)
                loss = self.loss_function(output, prediction)
                if visualize:
                    print("prediction:", prediction)
                    print("loss:", loss)
            else:
                loss = None
            return sample, output, loss
