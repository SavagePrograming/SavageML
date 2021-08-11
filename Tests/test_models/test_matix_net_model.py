import pickle

import numpy.random

from Tests import DATA_WIDTH, OUT_WIDTH, DATA_SIZE, TEST_DATA, TEST_XOR, LEARNING_RATE, PRECISION
from savageml.models import MatrixNetModel
from savageml.simulations import BinaryXorSimulation
import numpy as np

DIMENSIONS = [DATA_WIDTH, 10, OUT_WIDTH]
SIMULATION = BinaryXorSimulation((DATA_WIDTH,), DATA_SIZE)

EXPECTED_PARAMS = {
    "dimensions":DIMENSIONS
}

def test_predict_outputs_correct_size_ndarray():
    model = MatrixNetModel(**EXPECTED_PARAMS)
    prediction = model.predict(TEST_DATA)
    assert prediction.shape == (DATA_SIZE, OUT_WIDTH)


def test_predict_outputs_correct_size_simulation():
    model = MatrixNetModel(**EXPECTED_PARAMS)
    prediction = model.predict(SIMULATION)
    assert prediction.shape == (DATA_SIZE, OUT_WIDTH)


def test_fit_x_y():
    model = MatrixNetModel(**EXPECTED_PARAMS)
    model.fit(TEST_DATA, TEST_XOR, LEARNING_RATE)
    assert True


def test_fit_simulation():
    model = MatrixNetModel(**EXPECTED_PARAMS)
    model.fit(SIMULATION, learning_rate=LEARNING_RATE)
    assert True


def test_fit_simulation_score():
    model = MatrixNetModel(**EXPECTED_PARAMS)
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



def test_model_can_be_pickled():
    model = MatrixNetModel(**EXPECTED_PARAMS)
    data = pickle.dumps(model)
    new_model = pickle.loads(data)

    assert np.isclose(model.predict(TEST_DATA), new_model.predict(TEST_DATA), PRECISION).all()