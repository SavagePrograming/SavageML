from savageml.models.recursive_output_net_model import RecursiveOutputNetModel
from pynput import keyboard

from savageml.simulations.binary_sum_simulation import BinarySumSimulation
from savageml.simulations.simulation_simulation import SimulationSimulation
from savageml.utility.event_model_dataclasses import KeyEventData, build_wait_function_OR
from savageml.simulations.binary_and_sum_simulation import BinaryAndSumSimulation

STEPS = 10

model = RecursiveOutputNetModel((10,10, 10))
simulation = BinarySumSimulation(size=10, max_steps=10, model=model)
print("==================== Initial prediction ====================")
simulation.run(visualize=True)
print("==================== Starting Fit ====================")
model.fit(SimulationSimulation(BinarySumSimulation,
                               simulation_kwargs={"size":10, "max_steps":10, "model":model},
                               max_steps=10000), batch_size=100)
print("==================== Post fit prediction ====================")
simulation = BinarySumSimulation(size=10, max_steps=10, model=model)
simulation.run(visualize=True)