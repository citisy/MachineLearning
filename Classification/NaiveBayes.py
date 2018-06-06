# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import collections


class NB(object):
    def __init__(self, data, label):
        """
        :param data:
        :param label:
        :param rules: data每一个维度的规则
        """
        self.data = data
        self.label = label
        self.l = len(self.label)
        self.dict_ = collections.Counter(self.label)
        self.py_index = []
        self.py = []
        for (k, v) in self.dict_.items():
            self.py_index.append(np.where(self.label == k))
            self.py.append(self.l / v)

    def predict(self, data):
        p0 = []
        p1 = []
        for i in self.rules:
            if len(i) == 1:
                if data == i[0]:
                    p0.append(np.sum(data[self.zeros] == i[0]) / self.c0)
                    p1.append(np.sum(data[self.ones] == i[0]) / self.c1)
            else:
                if i[0] <= data < i[1]:
                    p0.append(np.sum(i[0] <= data[self.zeros] < i[1]) / self.c0)
                    p1.append(np.sum(i[0] <= data[self.ones] < i[1]) / self.c1)
        pc0 = self.pc0
        for i in p0:
            pc0 *= i
        pc1 = self.pc1
        for i in p1:
            pc1 *= i
        if pc0 > pc1:
            return 0
        return 1

    def gaussian_predict(self, x):
        self.sigma = []
        self.mu = []
        self.pyx = []
        for i in range(len(self.py_index)):
            mu_ = np.mean(self.data[self.py_index[i]], axis=0)
            # sigma_ = np.cov(self.data[self.py_index[i]])
            sigma_ = 0
            n = len(self.data[self.py_index[i]])
            for a in self.data[self.py_index[i]]:
                sigma_ += np.matmul(a - mu_, (a - mu_).T) / n
            pxy = np.exp(-0.5 * np.matmul(x - mu_, (x - mu_).T) / sigma_) / np.sqrt(2 * np.pi * sigma_)
            pyx = pxy * self.py[i]
            # print(sigma_)
            self.sigma.append(sigma_)
            self.mu.append(mu_)
            self.pyx.append(pyx)
        # print(self.py)
        # print(self.pyx)
        return np.argmax(self.pyx)

    def show(self):
        plt.clf()
        ax = plt.gca()
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.label)
        for i in range(len(self.py_index)):
            x = np.arange(self.mu[i][0] - 3, self.mu[i][0] + 3, 0.01)
            x1, x2 = np.meshgrid(x, x)
            y = np.exp(-0.5 * ((x1 - self.mu[i][0]) ** 2 + 0.5 * (x2 - self.mu[i][1]) ** 2))
            # print(y)
            # 等高线
            ax.contour(x1, x2, y)
            plt.draw()
        k = (self.mu[1][1] - self.mu[0][1]) / (self.mu[1][0] - self.mu[0][0])
        mid = [self.mu[1][0] + self.mu[0][0] / 2, self.mu[1][1] + self.mu[0][1] / 2]
        x = np.linspace(-5, 5)
        y = -(x - mid[0]) / k + mid[1]
        ax.plot(x, y)
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB

    # data = datasets.load_iris()
    # x = data.data
    # y = data.target
    # print(x)
    x, y = datasets.make_blobs(centers=2)
    model = NB(x, y)
    for i in range(len(x)):
        pre = model.gaussian_predict(x[i])
        print(pre, y[i])
    plt.figure()
    model.show()
    # m = GaussianNB().fit(x,y)
    # pre = m.predict(x)
    # for i in range(len(x)):
    #     print(pre[i], y[i])
