import numpy as np


class Norm(object):
    def __init__(self, data):
        """
        :param data: [n_sample, n_features]
        """
        self.data = data.copy()
        self.n_sample = np.shape(data)[0]
        self.n_features = np.shape(data)[1]

    def min_max(self):
        """
        liner changing, if a new data insert in, it will be define again.
        data fall in [0,1], use y = (x-min)/(max-min)
        if wanting to fall in [-1, 1], use y = (x-mean)/(max-min) or y = (y^2 - 1)
        of course, u can fall in any zones what u want.
        """
        min_ = np.min(self.data, axis=0)
        max_ = np.max(self.data, axis=0)
        self.data = (self.data - min_) / (max_ - min_)

    def z_score(self):
        """
        after normalization -> dimensionless
        data must fit with Gaussian distribution
        normalization function: y = (x - μ) / σ
        μ: mean of data, after normalization -> 0
        σ: standard deviation of data, after normalization -> 1
        """
        mean_ = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        self.data = (self.data - mean_) / std

    def binarizer(self, threshold=0.0):
        """
        data >= threshold -> 1
        data < threshold -> 0
        """
        for i in range(self.n_sample):
            for j in range(self.n_features):
                if self.data[i][j] >= threshold:
                    self.data[i][j] = 1
                else:
                    self.data[i][j] = 0

    def vec(self):
        """
        after normalization -> fall in the unit circle
        y = x / ||x||
        :return:
        """
        self.data /= np.linalg.norm(self.data, axis=1).reshape(-1, 1)

    def log_(self):
        """
        data must be greater than 1
        y = lg(x) / lg(max) or y = lg(x)
        :return:
        """
        max_ = np.max(self.data, axis=0)
        self.data = np.log10(self.data) / np.log10(max_)

    def arctan(self):
        self.data = np.arctan(self.data) * 2 / np.pi

    def fuzzy(self):
        """
        y = 0.5+0.5sin[pi/(max-min)*(x-0.5(max-min))]
        :return:
        """
        min_ = np.min(self.data, axis=0)
        max_ = np.max(self.data, axis=0)
        self.data = 0.5 + 0.5 * np.sin(np.pi / (max_ - min_) * (self.data - 0.5 * (max_ - min_)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = np.random.rand(100, 2) * 10
    norm = Norm(a)
    norm.fuzzy()
    plt.subplot(1, 2, 1)
    plt.scatter(a[:, 0], a[:, 1])
    plt.subplot(1, 2, 2)
    plt.scatter(norm.data[:, 0], norm.data[:, 1])
    plt.show()
