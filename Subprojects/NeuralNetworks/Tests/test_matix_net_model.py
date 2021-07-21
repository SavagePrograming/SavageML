import numpy.random

from Subprojects.NeuralNetworks.models import MatrixNetModel
from savageml.simulations import XorBinarySimulation
import numpy as np

DATA_SIZE = 100
DATA_WIDTH = 2
OUT_WIDTH = 1
DIMENSIONS = [DATA_WIDTH, 2, OUT_WIDTH]
SIMULATION = XorBinarySimulation((DATA_WIDTH,), DATA_SIZE)
RNG = numpy.random.default_rng(0)
TEST_DATA = RNG.choice([0.0, 1.0], (DATA_SIZE, DATA_WIDTH))

def test_predict_outputs_correct_size_ndarray():
    model = MatrixNetModel(DIMENSIONS)
    prediction = model.predict(TEST_DATA)
    assert prediction.shape == (DATA_SIZE, OUT_WIDTH)

def test_predict_outputs_correct_size_simulation():
    model = MatrixNetModel(DIMENSIONS)
    prediction = model.predict(SIMULATION)
    assert prediction.shape == (DATA_SIZE, OUT_WIDTH)


def test_fit():
    model = MatrixNetModel(DIMENSIONS)

    assert True



