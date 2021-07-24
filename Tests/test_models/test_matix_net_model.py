import numpy.random

from savageml.models import MatrixNetModel
from savageml.simulations import BinaryXorSimulation
import numpy as np

DATA_SIZE = 100
DATA_WIDTH = 2
OUT_WIDTH = 1
DIMENSIONS = [DATA_WIDTH, 10, OUT_WIDTH]
SIMULATION = BinaryXorSimulation((DATA_WIDTH,), DATA_SIZE)
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


def test_fit_simulation():
    model = MatrixNetModel(DIMENSIONS)
    model.fit(SIMULATION, learning_rate=LEARNING_RATE)
    assert True


def test_fit_simulation_score():
    model = MatrixNetModel(DIMENSIONS)
    simulation = BinaryXorSimulation((DATA_WIDTH,), 50000, seed=0)
    print()
    print(model.predict(np.array([[1.0, 0.0]])))
    print(model.predict(np.array([[0.0, 1.0]])))
    print(model.predict(np.array([[1.0, 1.0]])))
    print(model.predict(np.array([[0.0, 0.0]])))

    model.fit(simulation, learning_rate=LEARNING_RATE, batch_size=10)

    print(model.predict(np.array([[1.0, 0.0]])), model.predict(np.array([[1.0, 0.0]])) > .5)
    print(model.predict(np.array([[0.0, 1.0]])), model.predict(np.array([[0.0, 1.0]])) > .5)
    print(model.predict(np.array([[1.0, 1.0]])), model.predict(np.array([[1.0, 1.0]])) > .5)
    print(model.predict(np.array([[0.0, 0.0]])), model.predict(np.array([[0.0, 0.0]])) > .5)

    assert model.predict(np.array([[1.0, 0.0]])) > .75
    assert model.predict(np.array([[0.0, 1.0]])) > .75
    assert model.predict(np.array([[0.0, 0.0]])) < .25
    assert model.predict(np.array([[1.0, 1.0]])) < .25
