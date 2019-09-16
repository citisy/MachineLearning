# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class cluster:
    def __init__(self, data, k=3, itera=100, draw=0):
        self.data = np.array(data, dtype=float)
        self.k = k
        self.itera = itera
        self.draw = draw
        self.n_features = self.data.shape[1]  # 数据维度
        self.n_samples = self.data.shape[0]  # 数据数量
        self.cent_ind = np.array(range(self.n_samples), dtype=int)
        self.norm()
        self.train()

    def norm(self):
        """
        normalize the data
        """
        pass

    def train(self):
        """
        Attributes:
            cent_ind: 每一个点的所属簇类
        """
        pass

    # 求点与中心点的距离
    def get_r(self, x1, x2, method='euc'):
        # 欧氏距离
        if method == 'euc':
            return ((x1 - x2) ** 2).sum()
        # 余弦相似度
        elif method == 'cos':
            return -np.dot(x1, x2)/np.sqrt((x1**2).sum()*(x2**2).sum())

    def show(self, *args):
        """
        Visualization of algorithms.
        """
        pass

    def score(self):
        # Calinski-Harabasz score
        # 簇间协方差的迹
        bk = np.trace(np.cov(self.cent, rowvar=False))
        # 簇内协方差的迹
        cent_ = []
        wk = 0
        for i in range(self.k):
            for j in range(self.n_samples):
                if self.cent_ind[j] == i:
                    cent_.append(self.data[j])
            wk += np.trace(np.cov(cent_, rowvar=False))
        return bk * (self.n_samples - self.k) / (wk * (self.k - 1))
