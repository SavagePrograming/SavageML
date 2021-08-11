import numpy as np

from savageml.simulations import BinaryXorSimulation


def GET_RNG():
    return np.random.default_rng(0)


RNG = GET_RNG()
PRECISION = 0.0000001
LEARNING_RATE = 0.01

