import numpy

from ..utility import LossFunctions
import numpy as np
from savageml.simulations import BaseSimulation, SimulationState


class BinarySumSimulation(BaseSimulation):
    def __init__(self, size=1, max_steps=10, loss_function=LossFunctions.MSE, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.max_steps = max_steps
        self.step_count = 0
        self.loss_function = loss_function
        self.last_out = numpy.zeros((1, self.size))

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
            sample = self.random.choice([0.0, 1.], (1, self.size))
            if visualize:
                print("sample", sample)
            # print(self.last_out)
            s1 = self.last_out + sample
            c = np.roll((s1 > 1.0), 1, 1)
            # print(c)
            c[0][0] = 0.0
            s2 = 1.0 * (s1 == 1.0)
            s3 = s2 + c
            while np.any(s3 > 1.0):
                s1 = s3
                c = np.roll((s1 > 1.0), 1, 1)
                c[0][0] = 0.0
                s2 = 1.0 * (s1 == 1.0)
                s3 = s2 + c
            self.last_out = s3
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
