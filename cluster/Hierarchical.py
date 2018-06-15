# -*- coding: utf-8 -*-

"""
　(1) 将每个对象看作一类，计算两两之间的最小距离；
　(2) 将距离最小的两个类合并成一个新类；
　(3) 重新计算新类与所有类之间的距离；
　(4) 重复(2)、(3)，直到所有类最后合并成一类
"""

import numpy as np
import matplotlib.pyplot as plt
from cluster import cluster


class Hierarchical(cluster):
    def train(self):
        r = np.zeros((self.n_samples, self.n_samples))
        # 两点距离最大为根号2，设为10可视为忽略的点
        r += 10
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                r[i][j] = self.get_r(self.data[i], self.data[j])
        for _ in range(self.n_samples - self.k):
            min_ind = np.argmin(r)
            j = min_ind % self.n_samples
            i = min_ind // self.n_samples
            # 距离最近的簇只保留一个
            ind_j = np.where(self.cent_ind == self.cent_ind[j])
            ind_i = np.where(self.cent_ind == self.cent_ind[i])
            self.cent_ind[ind_j] = self.cent_ind[i]
            # 簇内成员距离都为10
            for i in ind_i[0]:
                for j in ind_j[0]:
                    r[min(i, j)][max(i, j)] = 10
            if self.draw:
                self.show_()
        print('train completed!')
        plt.show()

    # 二维可视化
    def show_(self):
        plt.clf()
        for i in range(self.n_features // 2):
            ax = plt.subplot(1, self.n_features // 2, i + 1)
            ax.scatter(self.data[:, i], self.data[:, i + 1], c=self.cent_ind)
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.cluster import AgglomerativeClustering

    X, y = datasets.make_moons(n_samples=1000, noise=0.05)
    # X, y = datasets.make_blobs()
    model = Hierarchical(X, 2, draw=1)
    # model = AgglomerativeClustering(n_clusters=2).fit(X)
