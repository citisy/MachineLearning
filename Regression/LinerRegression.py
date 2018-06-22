# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class Liner(object):
    def __init__(self, data, label):
        self.data = data.copy()
        self.label = label.copy()
        self.label = np.reshape(self.label, (-1, 1))
        self.n_samples = np.shape(data)[0]
        self.n_features = np.shape(data)[1]
        # 增加一维 -> x0 = 1
        self.data = np.concatenate((np.ones((self.n_samples, 1)), self.data), axis=1)

    def gradient_descent(self, lr=1e-3, itera=100):
        """
        :param lr: 学习率，太小会不收敛
        :param itera: 迭代次数，学习率越小，迭代次数越大
        :return:
        """
        xmean = np.mean(self.data, axis=0)
        ymean = np.mean(self.label)
        self.w = np.mat(ymean / xmean).T
        xMat = np.mat(self.data)
        yMat = np.mat(self.label)
        for i in range(itera):
            self.w += lr * (xMat.T * (yMat - xMat * self.w))

    def normal_equations(self):
        """
        w = inv(x'x)(x'y)
        """
        xMat = np.mat(self.data)
        yMat = np.mat(self.label)
        xTx = xMat.T * xMat
        self.w = xTx.I * (xMat.T * yMat)

    def predict(self, data):
        data_ = data.copy()
        data_ = np.concatenate((np.ones((len(data_), 1)), data_), axis=1)
        self.pre = data_ * self.w
        return self.pre

    def draw(self):
        plt.scatter(self.data[:, 1], self.label)
        x = np.linspace(-2, 2, 20).reshape(-1, 1)
        # x = np.mat(np.concatenate((np.ones((len(x), 1)), x.reshape(-1, 1)), axis=1))
        y = self.predict(x)
        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets

    x, y = datasets.make_regression(n_samples=100, n_features=1, random_state=0, noise=4.0,
                                    bias=100.0)
    model = Liner(x, y)
    model.gradient_descent()
    model.draw()
