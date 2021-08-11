import numpy as np

from savageml.utility.activation_functions import sigmoid


def test_sigmoid():
    print()
    for s in sigmoid(np.linspace(-10.0, 10.0, 50)):
        print(s)
    assert True
