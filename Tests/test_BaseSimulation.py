from savageml.Models.BaseModel import BaseModel
from savageml.Simulations.BaseSimulation import BaseSimulation
from savageml.Simulations import SimulationState


def test_initial_state_initialized():
    simulation = BaseSimulation()
    assert simulation.get_state() == SimulationState.INITIALIZED


def test_state_after_run_completed():
    simulation = BaseSimulation()
    simulation.run()
    assert simulation.get_state() == SimulationState.COMPLETE


def test_state_after_step_not_initialized():
    simulation = BaseSimulation()
    simulation.step()
    assert simulation.get_state() != SimulationState.INITIALIZED


def test_reset_state_initialized():
    simulation = BaseSimulation()
    simulation.step()
    simulation.reset()
    assert simulation.get_state() == SimulationState.INITIALIZED


def test_default_model_is_none():
    simulation = BaseSimulation()
    assert simulation.get_model() is None


def test_get_model():
    model = BaseModel()
    simulation = BaseSimulation(model=model)
    assert simulation.get_model() is model


def test_set_model():
    model = BaseModel()
    simulation = BaseSimulation()
    assert simulation.get_model() is None
    simulation.set_model(model)
    assert simulation.get_model() is model


def test_get_seed():
    simulation = BaseSimulation(seed=0)
    assert simulation.get_seed() == 0
    simulation = BaseSimulation(seed=1)
    assert simulation.get_seed() == 1


def test_set_seed():
    simulation = BaseSimulation()
    assert simulation.get_seed() != 0
    simulation.set_seed(0)
    assert simulation.get_seed() == 0
    simulation.set_seed(1)
    assert simulation.get_seed() == 1


def test_set_seed_random():
    simulation = BaseSimulation()
    simulation.set_seed(0)
    rngVal_first = simulation.random.random()
    rngVal_second = simulation.random.random()
    assert rngVal_first != rngVal_second

    simulation.set_seed(0)
    rngVal_set_again = simulation.random.random()
    assert rngVal_first == rngVal_set_again


def test_reset_random():
    simulation = BaseSimulation()
    rngVal_first = simulation.random.random()
    simulation.reset()
    rngVal_reset = simulation.random.random()
    assert rngVal_first == rngVal_reset
