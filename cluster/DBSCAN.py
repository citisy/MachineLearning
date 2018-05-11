#-*- coding: utf-8 -*-

'''
　(1) 设定扫描半径 Eps, 并规定扫描半径内的密度值。若当前点的半径范围内密度大于等于设定密度值，则设置当前点为核心点；若某点刚好在某核心点的半径边缘上，则设定此点为边界点；若某点既不是核心点又不是边界点，则此点为噪声点。
　(2) 删除噪声点。
　(3) 将距离在扫描半径内的所有核心点赋予边进行连通。
　(4) 每组连通的核心点标记为一个簇。
　(5) 将所有边界点指定到与之对应的核心点的簇总。
'''

import numpy as np
import matplotlib.pyplot as plt
from cluster import cluster

class DBSCAN(cluster):
    def __init__(self, data, draw=0, eps=0.01, threshold=3):
        self.eps = eps
        self.threshold = threshold
        super(DBSCAN, self).__init__(data, draw)

    # TODO 设置阈值
    def train(self):
        data = self.data.copy()
        self.cent_ind = np.array(range(self.col))
        for i in range(self.col):
            for j in range(i, self.col):
                r = self.get_r(data[i], data[j])
                if r <= self.eps:
                    # 距离最近的簇只保留一个
                    ind = np.where(self.cent_ind == self.cent_ind[max(i, j)])
                    self.cent_ind[ind] = self.cent_ind[min(i, j)]
        self.show_()

    def show_(self):
        plt.clf()
        for i in range(self.row // 2):
            ax = plt.subplot(1, self.row // 2, i + 1)
            ax.scatter(self.data[:, i], self.data[:, i + 1], c=self.cent_ind)
        plt.show()

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.cluster import AgglomerativeClustering
    X, y = datasets.make_moons(n_samples=500,noise=0.1)
    model = DBSCAN(X, draw=1)

