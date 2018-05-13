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
        super(DBSCAN, self).__init__(data, draw=draw)

    # 圈地运动，hhh
    def train(self):
        self.cent_ind = np.array(range(self.col), dtype=int)
        self.cent_cn = np.zeros(self.col, dtype=int)
        cent_ind = [[i] for i in range(self.col)]
        r = np.zeros((self.col, self.col)) + 10
        for i in range(self.col):
            for j in range(i+1, self.col):
                r_ = self.get_r(self.data[i], self.data[j])
                if r_ <= self.eps:
                    self.cent_cn[i] += 1
                    self.cent_cn[j] += 1
                    cent_ind[i].append(j)
                    r[i][j] = r_
                    r[j][i] = r_
        # core point
        core_point = []
        for i in range(self.col):
            if self.cent_cn[i] >= self.threshold:
                core_point.append(i)
                if len(cent_ind[i]) == 1:
                    continue
                ind_x = np.where(self.cent_ind == self.cent_ind[cent_ind[i][0]])
                for a in range(1, len(cent_ind[i])):
                    if r[i][cent_ind[i][a]] == 0 or r[cent_ind[i][a]][i] == 0:
                        continue
                    ind_y = np.where(self.cent_ind == self.cent_ind[cent_ind[i][a]])
                    # if ind_x is ind_y:
                    #     continue
                    ind = np.append(ind_x, ind_y)
                    self.cent_ind[ind_y] = self.cent_ind[cent_ind[i][0]]
                    # print(ind_x[0], ind_y[0], r[i][cent_ind[i][a]])
                    # print(ind)
                    # TODO 连通
                    # for x in range(len(ind)-1):
                    #     for y in range(x + 1, len(ind)):
                    #         r[ind[x]][ind[y]] = 0
                    #         r[ind[y]][ind[x]] = 0
                    for x in ind_x[0]:
                        for y in ind_y[0]:
                            r[x][y] = 0
                            r[y][x] = 0
                    if self.draw:
                        self.show_()


        # border point
        for i in range(self.col):
            if self.cent_cn[i] > 0 and self.cent_cn[i] < self.threshold:
                r_ = 10
                j_ = i
                # 没有归属点
                if min(r[i]) != 0:
                    # 查找最近的core point
                    for j in core_point:
                        r1 = self.get_r(self.data[i], self.data[j])
                        if r1 < r_:
                            r_ = r1
                            j_ = j
                self.cent_ind[i] = self.cent_ind[j_]
                if self.draw:
                    self.show_()
        # noise point
        for i in range(self.col):
            if self.cent_cn[i] == 0:
                self.cent_ind[i] = -1
        if self.draw:
            self.show_()
        print('train completed!')
        plt.show()

    def show_(self):
        plt.clf()
        for i in range(self.row//2):
            ax = plt.subplot(1, self.row//2, i+1)
            ax.scatter(self.data[:, i], self.data[:, i+1], c=self.cent_ind)
        plt.draw()
        plt.pause(0.0001)

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.cluster import AgglomerativeClustering
    X, y = datasets.make_moons(n_samples=500,noise=0.08)
    model = DBSCAN(X, draw=1, eps=0.005)

