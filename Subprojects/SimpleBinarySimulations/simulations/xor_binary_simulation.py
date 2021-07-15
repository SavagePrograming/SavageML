from savageml.simulations import BaseSimulation


class XorBinarySimulation(BaseSimulation):
    def __init__(self):
        super().__init__()

    def step(self, visualize=False) -> tuple:
        pass

    def __iter__(self):
        pass

    def reset(self):
        pass