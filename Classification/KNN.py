# -*- coding: utf-8 -*-


"""
knn是lazy learning，基本不学习，网络结构很简单，但每次都有遍历所有样本计算距离，所以计算量很大。
适合大规模数据，小数据错误率高。
判定标准：“近朱者赤，近墨者黑”以及“少数服从多数”。
"""
import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.colors import ListedColormap


class KNN(object):
    def __init__(self, data, label, k=10):
        self.data = data
        self.label = label
        self.k = k
        self.n_sample = len(self.data)

    def predict(self, x):
        n_test = len(x)
        pre = np.zeros(n_test, dtype=int)
        for a in range(n_test):
            r = np.zeros(self.n_sample)
            for i in range(self.n_sample):
                r[i] = np.linalg.norm(x[a] - self.data[i])
            argsort = r.argsort()
            # 统计预测点附近k个点的标签，取出现次数最多的标签
            dict_ = collections.Counter(self.label[argsort[:self.k]])
            maxv = 0
            for k, v in dict_.items():
                if maxv < v:
                    pre[a] = k
                    maxv = v
        return pre


if __name__ == '__main__':
    from sklearn import datasets

    x, y = datasets.make_blobs()
    model = KNN(x, y)
    pre = model.predict(x)
    for i in range(len(pre)):
        print(pre[i], y[i])
