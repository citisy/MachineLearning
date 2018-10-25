"""
　　(1) 任意选择k个对象作为初始的簇中心；
　　(2) repeat；
　　(3) 根据簇中对象的平均值，将每个对象(重新)赋予最类似的簇；
　　(4) 更新簇的平均值，即计算每个簇中对象的平均值；
　　(5) until不再发生变化。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cluster import cluster
import math
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)


class Kmeans(cluster):
    def norm(self):
        self.cent = np.zeros((self.k, self.n_features))
        for i in range(self.n_features):
            amax = self.data[:, i].max()
            amin = self.data[:, i].min()
            # 数据归一化
            self.data[:, i] /= amax
            # 随机生成中心点落在数据中
            self.cent[:, i] = np.random.random(self.k) * (amax - amin) / amax + amin / amax

    # 模型训练
    def train(self):
        if self.draw:
            self.ims = []
            self.col = math.ceil(np.sqrt(self.n_features/2))
            self.row = math.ceil(self.n_features/2/self.col)
            self.fig, self.ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)
            self.fig.set_tight_layout(True)

        for _ in range(self.itera):
            self.cent_cn = np.zeros(self.k, dtype=int)
            for a in range(self.n_samples):
                r = []
                # 求该点与所有中心点的距离
                for b in range(self.k):
                    r.append(self.get_r(self.data[a], self.cent[b]))
                # 离点最近的中心点的下标
                max_ind = np.argmin(r)
                # 每一点标记属于哪一个中心点
                self.cent_ind[a] = max_ind
                # 记录每个中心点有多少个附属点
                self.cent_cn[max_ind] += 1
            cent = np.zeros((self.k, self.n_features))
            # 遍历所有数据，更新中心点位置
            for a in range(self.n_samples):
                cent[self.cent_ind[a]] += self.data[a]
            for a in range(self.k):
                if self.cent_cn[a] == 0:
                    cent[a] = self.cent[a] + np.random.random(self.n_features) - 0.5
                else:
                    cent[a] /= self.cent_cn[a]
            # 中心点位置不变，训练完成，退出循环
            if (self.cent == cent).all():
                break
            self.cent = cent
            if self.draw:
                self.show(self.data, self.cent_ind, self.cent)
        print('train completed!')
        if self.draw:
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=2000 // len(self.ims), blit=True,
                                            repeat_delay=1000, repeat=False)
            # ani.save('../img/kmeans.gif', writer='pillow')
            plt.show()

    # 二维可视化
    def show(self, data, cent_ind, cent):
        im = []
        for i in range(self.n_features // 2):
            a = i // self.col
            b = i % self.col
            im.append(self.ax[a][b].scatter(data[:, i], data[:, i + 1], c=cent_ind, animated=True))
            im.append(self.ax[a][b].scatter(cent[:, i], cent[:, i + 1], c='r', animated=True))
        self.ims.append(im)


if __name__ == '__main__':
    from sklearn import datasets

    X, y = datasets.make_blobs(n_samples=500, n_features=2)
    model = Kmeans(X, 3, draw=1)
    # print(model.score())
    # y_pred = MiniBatchKMeans(n_clusters=2, batch_size=200, random_state=9).fit_predict(X)
