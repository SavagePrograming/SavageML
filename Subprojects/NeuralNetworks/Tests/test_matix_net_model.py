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
TEST_XOR = np.reshape((TEST_DATA[:, 0] != TEST_DATA[:, 1]), (DATA_SIZE, 1))
LEARNING_RATE = 0.01


def test_predict_outputs_correct_size_ndarray():
    model = MatrixNetModel(DIMENSIONS)
    prediction = model.predict(TEST_DATA)
    assert prediction.shape == (DATA_SIZE, OUT_WIDTH)


def test_predict_outputs_correct_size_simulation():
    model = MatrixNetModel(DIMENSIONS)
    prediction = model.predict(SIMULATION)
    assert prediction.shape == (DATA_SIZE, OUT_WIDTH)


def test_fit_x_y():
    model = MatrixNetModel(DIMENSIONS)
    model.fit(TEST_DATA, TEST_XOR, LEARNING_RATE)
    assert True
