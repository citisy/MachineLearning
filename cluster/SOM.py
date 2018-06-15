# -*- coding: utf-8 -*-

"""
SOM: Self-organizing Maps
适合环装数据的聚类
https://zhuanlan.zhihu.com/p/31637590
https://wenku.baidu.com/view/74927ae8aeaad1f346933f42.html?qq-pf-to=pcqq.c2c
https://blog.csdn.net/wj176623/article/details/52526617
"""

import numpy as np
import matplotlib.pyplot as plt
from cluster import cluster


class SOM(object):
    def __init__(self, data, lr=1, itera=10, batch_size=10, output_size=2):
        """
        :param data: input data
                    size -> [n_sample, input_size], input_size -> num of features
        :param lr: learning rate, will be change during training
        :param itera: max iteration
        :param batch_size: num of data when training
        :param output_size: num of classes
        """
        self.raw_data = data
        self.data = data.copy()
        self.lr = lr
        self.itera = itera
        self.batch_size = batch_size
        self.output_size = output_size
        self.n_sample = np.shape(data)[0]
        self.intput_size = np.shape(data)[1]
        self.w = np.random.rand(self.output_size, self.intput_size)
        self.output = np.zeros(self.n_sample)
        self.norm(self.data)
        self.train()

    def norm(self, data):
        for i in range(len(data)):
            data[i] /= np.linalg.norm(data[i])

    def train(self):
        t = 0
        for i in range(self.itera):
            n = self.getn(t)
            # self.update_lr(n)
            for j in range(self.n_sample):
                self.norm(self.w)
                self.d = np.matmul(self.w, self.data[j].T)
                argmax = np.argmax(self.d)
                neighbor = self.get_neighbor(argmax, n)
                for k, v in neighbor.items():
                    self.w[k] += self.update_lr(t, v) * (self.data[j] - self.w[k])
            t += 1

    def update_lr(self, t, n):
        return self.lr * np.exp(-(n * self.output_size)) / (t + 1)

    def getn(self, t):
        return (1 - t / self.itera)

    def get_neighbor(self, i, n):
        neighbor = {}
        for a in range(self.output_size):
            r = np.linalg.norm(self.w[a] - self.w[i])
            if r < n:
                neighbor[a] = r
        return neighbor

    def predict(self, data):
        self.test_data = data.copy()
        self.norm(self.test_data)
        n_sample = len(data)
        self.cent_ind = np.zeros(n_sample, dtype=int)
        d = np.matmul(self.w, self.test_data.T)
        self.cent_ind = np.argmax(d, axis=0)
        return self.cent_ind

    def show(self):
        plt.figure()
        plt.scatter(self.raw_data[:, 0], self.raw_data[:, 1], c=self.cent_ind)
        plt.figure()
        for i in range(self.intput_size // 2):
            ax = plt.subplot(1, self.intput_size // 2, i + 1)
            ax.scatter(self.data[:, i], self.data[:, i + 1], c=self.cent_ind)
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets

    x, y = datasets.make_blobs(centers=4)
    model = SOM(x, output_size=4, itera=10)
    # the num of prediction classes mill be less than output_size
    pre = model.predict(model.data)
    model.show()
