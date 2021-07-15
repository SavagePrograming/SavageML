import inspect

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

    @classmethod
    def _get_param_names(cls):
        """Get parameter names"""
        # Modified from scikit-learn base estimator
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("_get_param_names requires classes "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def __iter__(self):
        params = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            params[key] = value
        return self.__class__(**params)

    def __next__(self):
        if self.state == SimulationState.COMPLETE:
            raise StopIteration
        else:
            result = self.step()
            return result

    def set_seed(self, seed=get_default_seed()):
        self.seed = seed
        self.random = numpy.random.default_rng(self.seed)

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
        self.random = numpy.random.default_rng(self.seed)
        self.state = SimulationState.INITIALIZED
