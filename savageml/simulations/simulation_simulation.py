import numpy

from ..utility import LossFunctions
import numpy as np
from savageml.simulations import BaseSimulation, SimulationState


class SimulationSimulation(BaseSimulation):
    def __init__(self, simulation=None, simulation_kwargs=None, max_steps=10, loss_function=LossFunctions.MSE, **kwargs):
        super().__init__(**kwargs)
        if simulation_kwargs is None:
            self.simulation_kwargs = {}
        else:
            self.simulation_kwargs = simulation_kwargs
        self.simulation = simulation
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
            sample = self.simulation(**self.simulation_kwargs)
            if visualize:
                print("sample", sample)
            if self.model is not None:
                prediction = self.model.predict(sample)
                loss = self.loss_function(self.last_out, prediction)
                if visualize:
                    print("prediction:", prediction)
                    print("loss:", loss)
            else:
                loss = None
            return sample
