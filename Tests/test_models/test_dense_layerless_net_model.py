import pickle

from Tests import DATA_WIDTH, OUT_WIDTH, TEST_DATA, DATA_SIZE, XOR_SIMULATION, TEST_XOR, LEARNING_RATE, PRECISION
from savageml.models import LayerlessDenseNetModel
from savageml.simulations import BinaryXorSimulation
import numpy as np


HIDDEN_WIDTH = 10
EXPECTED_CONNECTION_COUNT = 88
EXPECTED_PARAMS = {
    "input_dimension": DATA_WIDTH,
    "hidden_dimension": HIDDEN_WIDTH,
    "output_dimension": OUT_WIDTH
}


def test_predict_outputs_correct_size_ndarray():
    model = LayerlessDenseNetModel(**EXPECTED_PARAMS)
    prediction = model.predict(TEST_DATA)
    assert prediction.shape == (DATA_SIZE, OUT_WIDTH)


def test_predict_outputs_correct_size_simulation():
    model = LayerlessDenseNetModel(**EXPECTED_PARAMS)
    prediction = model.predict(XOR_SIMULATION)
    assert prediction.shape == (DATA_SIZE, OUT_WIDTH)


def test_fit_x_y():
    model = LayerlessDenseNetModel(**EXPECTED_PARAMS)
    model.fit(TEST_DATA, TEST_XOR, LEARNING_RATE)
    assert True


def test_fit_simulation():
    model = LayerlessDenseNetModel(**EXPECTED_PARAMS)
    model.fit(XOR_SIMULATION, learning_rate=LEARNING_RATE)
    assert True


def test_from_connection_list():
    model = LayerlessDenseNetModel(**EXPECTED_PARAMS)
    model_clone = LayerlessDenseNetModel.from_connections_list(
        input_dimension=DATA_WIDTH,
        hidden_dimension=HIDDEN_WIDTH,
        output_dimension=OUT_WIDTH,
        connection_list=model.get_connections_list()
    )

    assert np.isclose(model.predict(TEST_DATA), model_clone.predict(TEST_DATA), PRECISION).all()


def test_get_connections_list():
    model = LayerlessDenseNetModel(**EXPECTED_PARAMS)
    assert len(model.get_connections_list()) == EXPECTED_CONNECTION_COUNT


def test_fit_simulation_score():
    model = LayerlessDenseNetModel(**EXPECTED_PARAMS)
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
    model = LayerlessDenseNetModel(**EXPECTED_PARAMS)
    data = pickle.dumps(model)
    new_model = pickle.loads(data)

    assert np.isclose(model.predict(TEST_DATA), new_model.predict(TEST_DATA), PRECISION).all()
