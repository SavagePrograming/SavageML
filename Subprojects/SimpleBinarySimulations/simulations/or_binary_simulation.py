from savageml.simulations import BaseSimulation


class OrBinarySimulation(BaseSimulation):
    def __init__(self):
        super().__init__()

    def step(self, visualize=False) -> tuple:
        pass

    def __iter__(self):
        pass

    def reset(self):
        pass