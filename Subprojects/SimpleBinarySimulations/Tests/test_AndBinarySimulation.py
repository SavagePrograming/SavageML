from savageml.simulations import SimulationState
from ..simulations import AndBinarySimulation


def test_initial_state_initialized():
    simulation = AndBinarySimulation()
    assert simulation.get_state() == SimulationState.INITIALIZED


def test_state_after_run_completed():
    simulation = AndBinarySimulation()
    simulation.run()
    assert simulation.get_state() == SimulationState.COMPLETE


def test_state_after_step_not_initialized():
    simulation = AndBinarySimulation()
    simulation.step()
    assert simulation.get_state() != SimulationState.INITIALIZED


def test_reset_state_initialized():
    simulation = AndBinarySimulation()
    simulation.step()
    simulation.reset()
    assert simulation.get_state() == SimulationState.INITIALIZED


def test_reflexive_simulation_duplicates():
    simulation = AndBinarySimulation(seed=1)
    simulation_duplicate = simulation.__iter__()
    assert simulation.get_seed() == simulation_duplicate.get_seed()
    assert simulation.step() == simulation_duplicate.step()


def test_reflexive_simulation_shape_correct():
    test_shape = (5, 7)
    simulation = AndBinarySimulation(seed=1, shape=test_shape)
    result = simulation.step()
    assert result[0].shape == test_shape
    assert result[1].shape == test_shape


def test_reflexive_simulation_reflexive():
    test_shape = (5, 7)
    simulation = AndBinarySimulation(seed=1, shape=test_shape)
    result = simulation.step()
    assert (result[0] == result[1]).all()

def test_reflexive_simulation_step_count():
    max_steps = 11
    simulation = AndBinarySimulation(seed=1, max_steps=max_steps)
    count = 0
    for _ in simulation:
        count += 1
    assert count == max_steps
    simulation.reset()
    assert simulation.get_state() == SimulationState.INITIALIZED
    for _ in range(max_steps):
        simulation.step()
    assert simulation.get_state() == SimulationState.COMPLETE
