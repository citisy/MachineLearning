# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../Regression')
from LinerRegression import Liner


class Logistic(object):
    def __init__(self, data, label):
        self.data = data.copy()
        self.label = label.copy()
        self.n_samples = np.shape(data)[0]
        self.n_features = np.shape(data)[1]
        self.train()

    def train(self):
        self.model = Liner(self.data, self.label)
        self.model.normal_equations()

    def predict(self, data):
        z = self.model.predict(data)
        self.sigma = 1 / (1 + np.exp(-z))
        self.per = np.zeros(self.n_samples, dtype=int)
        for i in range(self.n_samples):
            if self.sigma[i] >= 0.5:
                self.per[i] = 1
        self.draw(data)

    def draw(self, data):
        plt.scatter(data[:, 0], data[:, 1], c=self.per)
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets

    x, y = datasets.make_blobs(centers=2)
    model = Logistic(x, y)
    model.predict(x)
