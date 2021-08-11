from Tests import RNG, np, BinaryXorSimulation

DATA_SIZE = 100
DATA_WIDTH = 2
OUT_WIDTH = 1
TEST_DATA = RNG.choice([0.0, 1.0], (DATA_SIZE, DATA_WIDTH))

TEST_XOR = np.reshape((TEST_DATA[:, 0] != TEST_DATA[:, 1]), (DATA_SIZE, 1))
XOR_SIMULATION = BinaryXorSimulation((DATA_WIDTH,), DATA_SIZE)