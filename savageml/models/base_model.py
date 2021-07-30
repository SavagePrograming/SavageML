from typing import Union, Iterable

import numpy as np
from sklearn.base import BaseEstimator
from numpy import ndarray


class BaseModel(BaseEstimator):
    """The Base Model Class

    SavageML Models are implementations of the scikit-learn estimator and predictor.
    BaseModel is a base class showing what functions every model needs to implement.

    Parameters
    ----------
    test_param: bool, optional
        A test parm allowing the tests for the base model to work. Defaults to True

    """

    test_param: bool

    def __init__(self, test_param=True):
        """Constructor Method"""
        self.test_param = test_param
        pass

    def clone(self):
        """Creates an exact copy of the model, in it's initial state

        Returns
        -------
        BaseModel
            The cloned copy of the class
        """
        instance = self.__class__(**self.get_params())
        return instance

    def fit(self,
            x: Union[np.ndarray, Iterable],
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

    def predict(self, x: Union[np.ndarray, Iterable]) -> ndarray:
        """All models must have a predict function. This function must take an np.ndarray or
         an iterable that produces (x, ...) tuples. The function must then predict y values based on the input.

        Parameters
        ----------
        x: np.ndarray, Iterable
            The input values to the model, or an iterable that produces (x, ...) tuples.

        Returns
        -------
        np.ndarray
            The predicted output

        """
        pass
