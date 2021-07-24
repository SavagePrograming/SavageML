from ..utility import LossFunctions
from savageml.simulations import BaseSimulation, SimulationState


class BinaryReflexiveSimulation(BaseSimulation):
    def __init__(self, shape=(1,), max_steps=10, loss_function=LossFunctions.MSE, **kwargs):
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
            if self.model is not None:
                prediction = self.model.predict(sample)
                loss = self.loss_function(sample, prediction)
            else:
                loss = None
            return sample, sample.copy(), loss
