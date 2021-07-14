import random
import time

from ..models import BaseModel
from ..simulations import SimulationState


def get_default_seed():
    return int(time.time() * 1E6)


class BaseSimulation:
    def __init__(self, model=None, seed=get_default_seed()):
        self.seed: int = seed
        self.model: BaseModel = model
        self.state: SimulationState = SimulationState.INITIALIZED
        self.random: random.Random = random.Random()
        self.random.seed(self.seed)

    def __iter__(self):
        return self.__class__(self.model, self.seed)

    def __next__(self):
        if self.state == SimulationState.COMPLETE:
            raise StopIteration
        else:
            result = self.step()
            return result

    def set_seed(self, seed=get_default_seed()):
        self.seed = seed
        self.random.seed(self.seed)

    def get_seed(self):
        return self.seed

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def get_state(self):
        return self.state

    def step(self, visualize=False) -> tuple:
        self.state = SimulationState.COMPLETE
        return ()

    def run(self, visualize=False):
        while not self.get_state() == SimulationState.COMPLETE:
            self.step(visualize=visualize)

    def reset(self):
        self.random.seed(self.seed)
        self.state = SimulationState.INITIALIZED
