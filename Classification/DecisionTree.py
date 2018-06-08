# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import collections

"""
id3
c4.5
cart
"""


class DT(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.n_samples = len(self.label)
        self.n_i = len(np.unique(self.data))
        dict_ = collections.Counter(self.label)
        self.dict_ = {}
        for (k, v) in dict_.items():
            self.dict_[k] = {}
            self.dict_[k]['index'] = np.where(self.label == k)
            self.dict_[k]['count'] = v
            self.dict_[k]['py'] = v / self.n_samples
        self.n_features = len(self.dict_)

    def id3(self, x):
        self.get_entropy()
        return self.make_trees(x)

    def c45(self):
        pass

    def cart(self):
        pass

    def get_entropy(self):
        n_features = len(self.data[0])
        np.unique(self.data, axis=0)
        info_d = 0
        for k in self.dict_.keys():
            info_d -= self.dict_[k]['py'] * np.log2(self.dict_[k]['py'])
        self.gain_i = []
        for a in range(n_features):
            features = np.unique(self.data[:, a])
            info_i = 0
            for i in features:
                info_di = 0
                for k in self.dict_.keys():
                    n = np.sum(self.data[self.dict_[k]['index'], a] == i)
                    if n == 0:
                        continue
                    pi = n / self.dict_[k]['count']
                    info_di -= pi * np.log2(pi)
                p = np.sum(self.data[:, a] == i) / self.n_samples
                info_i += p * info_di
            self.gain_i.append(info_d - info_di)
        self.sort = [i for i in range(n_features)]
        # 冒泡排序
        for i in range(n_features):
            for j in range(i, n_features):
                if self.gain_i[i] < self.gain_i[j]:
                    self.gain_i[i], self.gain_i[j] = self.gain_i[j], self.gain_i[i]
                    self.sort[i], self.sort[j] = self.sort[j], self.sort[i]

    def make_trees(self, x):
        n_sample = len(x)
        pre = np.zeros(n_sample, dtype=int)
        for a in range(n_sample):
            index = range(self.n_samples)
            for b in self.sort:
                if x[a][b] in self.data[:, b]:
                    print(x[a][b], self.data[index])
                    index = np.where(self.data[index][b] == x[a][b])
                    print(index)
            # print(index)
            pre[a] = self.label[index]
        return pre



if __name__ == '__main__':
    x = np.random.randint(5, size=(100, 4))
    y = np.random.randint(5, size=(100,))

    model = DT(x, y)
    pre = model.id3(x)
    for i in range(100):
        print(pre[i], y[i])
