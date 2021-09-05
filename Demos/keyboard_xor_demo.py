from savageml.models.event_model import EventModel
from pynput import keyboard
from savageml.utility.event_model_dataclasses import KeyEventData, build_wait_function_OR
from savageml.simulations.binary_xor_simulation import BinaryXorSimulation

STEPS = 10

events = [KeyEventData(keyboard.KeyCode.from_char("1"), released=True)]
model = EventModel(events, wait_function=build_wait_function_OR(events, events), wait_timeout=3.0)
simulation = BinaryXorSimulation(max_steps=10, model=model)
simulation.run(visualize=True)
model.clean()