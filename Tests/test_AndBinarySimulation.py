from savageml.simulations import SimulationState
from savageml.simulations import AndBinarySimulation


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


def test_simulation_duplicates():
    simulation = AndBinarySimulation(seed=1)
    simulation_duplicate = simulation.__iter__()
    assert simulation.get_seed() == simulation_duplicate.get_seed()
    assert simulation.step() == simulation_duplicate.step()


def test_and_simulation_shape_correct():
    test_shape = (5, 7)
    expected_out_shape = (1,)
    simulation = AndBinarySimulation(seed=1, shape=test_shape)
    result = simulation.step()
    assert result[0].shape == test_shape
    assert result[1].shape == expected_out_shape


def test_and_simulation_all_true():
    test_shape = (2, 2)
    all_ones_seeds = 4
    simulation = AndBinarySimulation(seed=all_ones_seeds, shape=test_shape)
    result = simulation.step()
    assert (result[0] == 1.0).all()
    assert (result[1] == 1.0).all()


def test_and_simulation_all_false():
    test_shape = (2, 2)
    all_zero_seeds = 36
    simulation = AndBinarySimulation(seed=all_zero_seeds, shape=test_shape)
    result = simulation.step()
    assert (result[0] == 0.0).all()
    assert (result[1] == 0.0).all()


def test_and_simulation_50_50():
    test_shape = (2, 2)
    _50_50_seeds = 31
    simulation = AndBinarySimulation(seed=_50_50_seeds, shape=test_shape)
    result = simulation.step()
    assert (result[0] == 1.0).sum() == (result[0] == 0.0).sum()
    assert (result[1] == 0.0).all()


def test_simulation_step_count():
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
