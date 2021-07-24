from sklearn.base import BaseEstimator
from numpy import ndarray


class BaseModel(BaseEstimator):
    """
    BaseModel:

    """

    test_param: bool

    def __init__(self, test_param=True):
        """

        :param test_param:
        """
        self.test_param = test_param
        pass

    def clone(self):
        """

        :return:
        """
        instance = self.__class__()
        instance.set_params(**self.get_params())
        return instance

    def fit(self, x, y=None):
        """

        :param x:
        :param y:
        :return:
        """
        pass

    def predict(self, x: ndarray) -> ndarray:
        """

        :param x:
        :return:
        """
        pass
