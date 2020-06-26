import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GaussianFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-1. / 2. * np.sum(arg ** 2, axis))

    def fit(self, x, y=None):
        self.centers_ = np.linspace(x.min(), x.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, x):
        return self._gauss_basis(x[:, :, np.newaxis],
                                 self.centers_,
                                 self.width_,
                                 axis=1)
