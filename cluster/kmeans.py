#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Kmeans(object):
    def __init__(self, data, k, itera=10, draw=0):
        self.data = np.array(data)
        self.k = k
        self.itera = itera
        self.draw = draw
        self.init_cent()
        self.train()


    # 初始化中心点位置
    def init_cent(self):
        self.cent = np.zeros((self.k, np.shape(self.data)[1]))
        for i in range(np.shape(self.data)[1]):
            amax = self.data[:, i].max()
            amin = self.data[:, i].min()
            # 数据归一化
            # self.data = (self.data-amin)/(amax-amin)
            self.data[:, i] /= amax
            # 随机生成中心点落在数据中
            self.cent[:, i] = np.random.random(self.k)*(amax-amin)/amax + amin/amax

    # 模型训练
    def train(self):
        for _ in range(self.itera):
            self.cent_ind = ['' for _ in range(np.shape(self.data)[0])]
            self.cent_cn = [0 for _ in range(self.k)]
            for a in range(np.shape(self.data)[0]):
                r = []
                # 求该点与所有中心点的距离
                for b in range(self.k):
                    r.append(((self.data[a]-self.cent[b])**2).sum())
                # 离点最近的中心点的下标
                max_ind = np.argmin(r)
                # 每一点标记属于哪一个中心点
                self.cent_ind[a] = max_ind
                # 记录每个中心点有多少个附属点
                self.cent_cn[max_ind] += 1
            cent = np.zeros((self.k, np.shape(self.data)[1]))
            for a in range(np.shape(self.data)[0]):
                cent[self.cent_ind[a]] += self.data[a]/self.cent_cn[self.cent_ind[a]]
            for a in range(self.k):
                if self.cent_cn[a] == 0:
                    continue
                else:
                    self.cent[a] = cent[a]
            if self.draw:
                self.show_()


    # 二维可视化
    def show_(self):
        plt.clf()
        for i in range(np.shape(self.data)[1]//2):
            ax = plt.subplot(1, np.shape(self.data)[1]//2, i+1)
            ax.scatter(self.data[:, i], self.data[:, i+1], c=self.cent_ind)
            ax.scatter(self.cent[:, i], self.cent[:, i+1], c='r')
        plt.draw()
        plt.pause(1)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.cluster import MiniBatchKMeans
    data = datasets.load_iris()  # sklearn内置的鸢尾花数据集
    X = data.data
    model = Kmeans(X, 2, draw=1, itera=100)
    y_pred = MiniBatchKMeans(n_clusters=2, batch_size=200, random_state=9).fit_predict(X)