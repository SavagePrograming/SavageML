from enum import Enum


class SimulationState(Enum):
    COMPLETE = 0
    INITIALIZED = 1
    RUNNING = 2