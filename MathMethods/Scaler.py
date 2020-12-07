import numpy as np


class Scaler:
    def min_max(self, data, mi=None, ma=None):
        """y = (x-min)/(max-min)
        liner changing, if a new data insert in, it will be define again.
        data fall in [0,1]"""
        data = np.array(data)
        mi = mi if mi is not None else np.min(data, axis=0)
        ma = ma if ma is not None else np.max(data, axis=0)
        data = (data - mi) / (ma - mi)
        return data, (mi, ma)

    def mean(self, data, mu=None, mi=None, ma=None):
        """y = (x-mean)/(max-min)
        fall in [-1, 1]"""
        data = np.array(data)
        mu = mu if mi is not None else np.mean(data, axis=0)
        mi = mi if mi is not None else np.min(data, axis=0)
        ma = ma if ma is not None else np.max(data, axis=0)
        data = (data - mu) / (ma - mi)
        return data, (mu, mi, ma)

    def z_score(self, data, mu=None, std=None):
        """y = (x - μ) / σ
        after normalization -> dimensionless
        data must fit with Gaussian distribution
        normalization function: y = (x - μ) / σ
        μ: mean of data, after normalization -> 0
        σ: standard deviation of data, after normalization -> 1
        """
        data = np.array(data)
        mu = mu if mu is not None else np.mean(data, axis=0)
        std = std if std is not None else np.std(data, axis=0)
        data = (data - mu) / std
        return data, (mu, std)

    def vec(self, data, axis=None):
        """y = x / ||x||
        after normalization -> fall in the unit circle
        """
        data = np.array(data)
        data /= np.linalg.norm(data, axis=axis, keepdims=True)
        return data

    def log(self, data, ma=None):
        """y = lg(x) / lg(max(x)) or y = lg(x)
        data must be greater than 1
        """
        data = np.array(data)
        ma = ma if ma is not None else np.max(data, axis=0)
        data = np.log10(data) / np.log10(ma)
        return data, ma

    def arctan(self, data):
        """y = arctan(x) * 2 / pi"""
        data = np.array(data)
        data = np.arctan(data) * 2 / np.pi
        return data

    def binarizer(self, data, threshold=None):
        """
        data >= threshold -> 1
        data < threshold -> 0
        """
        data = np.array(data)
        threshold = threshold if threshold is not None else np.mean(data, axis=0)
        norm = np.zeros_like(data)
        norm[np.where(data >= threshold)] = 1
        return norm

    def fuzzy(self, data, mi=None, ma=None):
        """y = 0.5+0.5sin[pi/(max-min)*(x-0.5(max-min))]"""
        data = np.array(data)
        mi = mi or np.min(data, axis=0)
        ma = ma or np.max(data, axis=0)
        data = 0.5 + 0.5 * np.sin(np.pi / (ma - mi) * (data - 0.5 * (ma - mi)))
        return data, (mi, ma)


scaler = Scaler()
