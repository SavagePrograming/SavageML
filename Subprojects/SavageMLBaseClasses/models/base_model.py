import json

from sklearn.base import BaseEstimator
from numpy import ndarray


class BaseModel(BaseEstimator):
    def __init__(self, test_param=True):
        self.test_param = test_param
        pass

    def clone(self):
        instance = self.__class__()
        instance.set_params(**self.get_params())
        return instance

    def fit(self, x, y=None):
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass
