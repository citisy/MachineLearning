"""
　(1) 设定扫描半径 Eps, 并规定扫描半径内的密度值。若当前点的半径范围内密度大于等于设定密度值，则设置当前点为核心点；若某点刚好在某核心点的半径边缘上，则设定此点为边界点；若某点既不是核心点又不是边界点，则此点为噪声点。
　(2) 删除噪声点。
　(3) 将距离在扫描半径内的所有核心点赋予边进行连通。
　(4) 每组连通的核心点标记为一个簇。
　(5) 将所有边界点指定到与之对应的核心点的簇总。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cluster import cluster
import time
import math
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)


class DBSCAN(cluster):
    def __init__(self, data, draw=0, eps=0.01, threshold=3):
        self.eps = eps
        self.threshold = threshold
        super(DBSCAN, self).__init__(data, draw=draw)

    def norm(self):
        """
        before norm:
        >> data
        >>[[-3.62194721 -5.49173113]
         [-3.23435367 -4.67226512]
         [-1.58990744 -9.87007247]
         [ 1.95358937 -1.92006285]
         [ 2.68055418 -1.53837307]]
        after norm
        >>data
        >>[[-0.83333333 -0.46366859]
         [-0.74415627 -0.39448082]
         [-0.36580402 -0.83333333]
         [ 0.44947953 -0.16211151]
         [ 0.61673874 -0.12988532]]
        all data will fall between [-1, 1]
        """
        for i in range(self.n_features):
            amax = abs(self.data[:, i].max())
            amin = abs(self.data[:, i].min())
            self.data[:, i] /= max(amax, amin) * 1.2

    def train(self):
        """
        eg:
            index: list type
                    [0,1,2] [3,4,5]
            distances: list type
                       10      0.4
                      0.4      10
             there is 2 class according to the list of distances,
             0,1,2 is in a class and 3,4,5 is in another class,
             we know that, the list of distances is symmetry,
             u can only use half of list distances
             for changing the code by yourselves, it's very easy(smile).
        """
        if self.draw:
            self.ims = []
            self.col = math.ceil(np.sqrt(self.n_features / 2))
            self.row = math.ceil(self.n_features / 2 / self.col)
            self.fig, self.ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)
            self.fig.set_tight_layout(True)

        stime = time.time()
        distances = [[10 for _ in range(self.n_samples)] for __ in range(self.n_samples)]
        index = [[i] for i in range(self.n_samples)]

        self.cent_cn = np.zeros(self.n_samples, dtype=int)
        cent_ind = [[] for _ in range(self.n_samples)]
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                r = self.get_r(self.data[i], self.data[j])
                distances[i][j] = r
                if r <= self.eps:
                    self.cent_cn[i] += 1
                    self.cent_cn[j] += 1
                    cent_ind[i].append(j)

        # core point
        core_point = []
        for i in range(self.n_samples):
            if self.cent_cn[i] >= self.threshold:
                core_point.append(i)
                if len(cent_ind[i]) == 0:
                    continue

                for j in cent_ind[i]:
                    x = y = -1
                    for k in range(len(index)):  # find data[i] and data[j] in distances' index
                        if i in index[k]:
                            x = k
                        if j in index[k]:
                            y = k

                    if all([x == y, x != -1, y != -1]):  # x and y in the same class
                        continue

                    index[x] += index[y]
                    for k in index[x]:
                        self.cent_ind[k] = i  # we label the class with data[i]'s index

                    for k in range(len(distances)):
                        if x == k:
                            continue
                        distances[x][k] = min(distances[x][k], distances[y][k])

                    for k in range(len(distances)):
                        del distances[k][y]

                    del distances[y]
                    del index[y]

                    if self.draw:
                        self.show(self.data, self.cent_ind)

        # border point
        for i in range(self.n_samples):
            if 0 < self.cent_cn[i] < self.threshold:
                x = -1
                for k in range(len(index)):
                    if i in index[k]:
                        x = k
                if x != -1 and len(index[x]) > 1:  # it is core point class
                    continue
                argsort = np.argsort(distances[x])  # sort by distances to other class
                j = 0
                while j < len(index):
                    y = argsort[j]
                    if len(index[y]) > 1:  # find the core point class
                        index[x] += index[y]
                        for k in index[x]:
                            self.cent_ind[k] = i

                        for k in range(len(distances)):
                            distances[x][k] = min(distances[x][k], distances[y][k])

                        for k in range(len(distances)):
                            del distances[k][y]

                        del distances[y]
                        del index[y]
                        break
                    j += 1
        if self.draw:
            self.show(self.data, self.cent_ind)

        # noise point
        for i in range(self.n_samples):
            if self.cent_cn[i] == 0:
                self.cent_ind[i] = -1
        if self.draw:
            self.show(self.data, self.cent_ind)

        etime = time.time()
        print('train completed! time: %s' % str(etime - stime))
        if self.draw:
            print(len(self.ims))
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=2000 / len(self.ims), blit=True,
                                            repeat_delay=0, repeat=True)
            ani.save('../img/DBSCAN.gif', writer='imagemagick')
            plt.show()

    def show(self, data, cent_ind):
        im = []
        for i in range(self.n_features // 2):
            a = i // self.col
            b = i % self.col
            im.append(self.ax[a][b].scatter(data[:, i], data[:, i + 1], c=cent_ind, animated=True))
        self.ims.append(im)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.cluster import AgglomerativeClustering

    np.random.seed(6)
    X, y = datasets.make_moons(n_samples=500, noise=0.08)
    model = DBSCAN(X, draw=1, eps=0.01)
