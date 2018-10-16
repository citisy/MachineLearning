# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from Regression.LinerRegression import Liner


class Logistic(object):
    def __init__(self, data, label):
        self.data = np.array(data, dtype=float)
        self.label = np.array(label)
        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self.train()

    def train(self):
        self.model = Liner(self.data, self.label)
        self.model.normal_equations()

    def predict(self, data):
        data = np.array(data, dtype=float)
        n_samples = data.shape[0]
        z = self.model.predict(data)
        sigma = 1 / (1 + np.exp(-z))
        pre = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if sigma[i] >= 0.5:
                pre[i] = 1
        self.show(data, pre, sigma)

    def show(self, data, pre, sigma):
        plt.scatter(data[:, 0], data[:, 1], c=pre)
        plt.plot(data[:, 0], sigma)
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets

    x, y = datasets.make_regression(n_samples=100, n_features=2, random_state=0, noise=4.0,
                                    bias=100.0)
    x = [[i, 0] for i in range(50)]
    x += [[i, 1] for i in range(50, 100)]
    y = [0 for i in range(50)]
    y += [1 for i in range(50)]

    model = Logistic(x, y)
    model.predict(x)
