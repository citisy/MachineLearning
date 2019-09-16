"""
　(1) 将每个对象看作一类，计算两两之间的最小距离；
　(2) 将距离最小的两个类合并成一个新类；
　(3) 重新计算新类与所有类之间的距离；
　(4) 重复(2)、(3)，直到所有类最后合并成一类
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cluster import cluster
import time
import math
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)

class Hierarchical(cluster):
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
        if self.draw:
            self.ims = []
            self.col = math.ceil(np.sqrt(self.n_features/2))
            self.row = math.ceil(self.n_features/2/self.col)
            self.fig, self.ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)

        stime = time.time()
        distances = np.zeros((self.n_samples, self.n_samples))
        # 两点距离最大为根号2，设为10可视为忽略的点
        distances += 10

        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                distances[i][j] = self.get_r(self.data[i], self.data[j])

        for _ in range(self.n_samples - self.k):
            min_ind = np.argmin(distances)
            j = min_ind % self.n_samples
            i = min_ind // self.n_samples
            # 距离最近的簇只保留一个
            ind_j = np.where(self.cent_ind == self.cent_ind[j])
            ind_i = np.where(self.cent_ind == self.cent_ind[i])
            self.cent_ind[ind_j] = self.cent_ind[i]
            # 簇内成员距离都为10
            for i in ind_i[0]:
                for j in ind_j[0]:
                    distances[min(i, j)][max(i, j)] = 10

            if self.draw:
                if _ % 5 == 4:
                    self.show(self.data, self.cent_ind)

        etime = time.time()
        print('train completed! time: %s' % str(etime - stime))
        if self.draw:
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=1000 / len(self.ims), blit=True,
                                            repeat_delay=1000, repeat=False)
            # ani.save('../img/Hierarchical.gif', writer='pillow')
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
    X, y = datasets.make_moons(n_samples=502, noise=0.08)
    # X, y = datasets.make_blobs(n_samples=100, n_features=4)
    model = Hierarchical(X, 2, draw=1)
    # model = AgglomerativeClustering(n_clusters=2).fit(X)
