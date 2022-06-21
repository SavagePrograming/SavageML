from typing import Union, Iterable, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from numpy import ndarray


class RecursiveModel(BaseEstimator):
    """A Base Recursive Model Class

   The Recursive Model shows

    Parameters
    ----------
    test_param: bool, optional
        A test parm allowing the tests for the base model to work. Defaults to True

    """

    state: object

    def __init__(self, state=None):
        """Constructor Method"""
        self.state = state

    def fit(self,
            x: Union[np.ndarray, Iterable[np.ndarray], Iterable[Iterable]],
            y: np.ndarray = None
             ) -> None:
        """ All models must have a fit function. The fit function must support two modes:

        In the first mode, `x` and `y` are given as separate inputs, where each is an :class:`np.ndarray` .
        The first index of both arrays must be the batch index.

        In the second mode only `x` is given and it is expected to be an :class:`Iterable` . That iterable should
        produce tuples where the 0th and 1st index are both :class:`np.ndarray` representing x and y respectively

        In either case the fit function has the model learn how to predict `y` from `x`

        Parameters
        ----------
        x: np.ndarray, Iterable
            The input values to the model, or an iterable that produces (x, y, ...) tuples.

        y: np.ndarray, optional
            The expected output values if an iterable is not provided.

        """
        pass

    def predict(self, x: Union[np.ndarray, Iterable], state=None) -> ndarray:
        """All models must have a predict function. This function must take an np.ndarray or
         an iterable that produces (x, ...) tuples. The function must then predict y values based on the input.

        Parameters
        ----------
        x: np.ndarray, Iterable
            The input values to the model, or an iterable that produces (x, ...) tuples.
        state:
            The hidden state of the model.

        Returns
        -------
        np.ndarray, Tuple
            The predicted output, or a tuple with the output and the hidden state

        """
        pass

    def init_hidden_state(self):
        return None

    def set_hidden_state(self, state: object):
        self.state = state

    def get_hidden_state(self) -> object:
        return self.state
