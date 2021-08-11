import numpy.random
import time

from typing import Tuple, List, Union

from ..models import BaseModel
from .simulation_state import SimulationState

test = numpy.random.default_rng(10)

def get_default_seed():
    return int(time.time() * 1E6)


class BaseSimulation:
    def __init__(self, model=None, seed=get_default_seed()):
        self.seed: int = seed
        self.model: BaseModel = model
        self.state: SimulationState = SimulationState.INITIALIZED
        self.random: numpy.random.Generator = numpy.random.default_rng(self.seed)

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

    def step(self, visualize=False) -> Tuple:
        self.state = SimulationState.COMPLETE
        return ()

    def run(self, visualize=False):
        while not self.get_state() == SimulationState.COMPLETE:
            self.step(visualize=visualize)

    def reset(self):
        self.random.seed(self.seed)
        self.state = SimulationState.INITIALIZED
