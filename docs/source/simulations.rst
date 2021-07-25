Simulations
===========
Simulations are another core feature of the savageml library.
Simulations are iterators that produce data that a model can use for learning.
Simulations also take as model as input, which can control the flow of the simulation.
For example, a simulation of mario would use the model to make marios moves.
Simulations can have more than one model.

When looped through simulations should return a tuple.
If there is one model in use, the simulation should return at tuple similar to:
(input_to_model, expected_output, score).
If there is more than one model, should return a tuple with
the (input_to_model, expected_output, score) tuple for each of the models.

.. toctree::
    simulations/BaseSimulation.rst
    simulations/BinaryAndSimulation
    simulations/BinaryOrSimulation
    simulations/BinaryReflexiveSimulation
    simulations/BinaryXorSimulation
    simulations/SimulationState