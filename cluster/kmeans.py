#-*- coding: utf-8 -*-

'''
　　(1) 任意选择k个对象作为初始的簇中心；
　　(2) repeat；
　　(3) 根据簇中对象的平均值，将每个对象(重新)赋予最类似的簇；
　　(4) 更新簇的平均值，即计算每个簇中对象的平均值；
　　(5) until不再发生变化。
'''

import numpy as np
import matplotlib.pyplot as plt
from cluster import cluster

class Kmeans(cluster):
    # 模型训练
    def train(self):
        self.cent_ind = np.zeros(self.col, dtype=int)
        for _ in range(self.itera):
            self.cent_cn = np.zeros(self.k, dtype=int)
            for a in range(self.col):
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
            cent = np.zeros((self.k, self.row))
            # 遍历所有数据，更新中心点位置
            for a in range(self.col):
                cent[self.cent_ind[a]] += self.data[a]
            for a in range(self.k):
                if self.cent_cn[a] == 0:
                    cent[a] = self.cent[a] + np.random.random(self.row) - 0.5
                else:
                    cent[a] /= self.cent_cn[a]
            # 中心点位置不变，训练完成，退出循环
            if (self.cent == cent).all():
                break
            self.cent = cent
            if self.draw:
                self.show_()
        print('train completed!')
        if self.draw:
            plt.show()

    # 二维可视化
    def show_(self):
        plt.clf()
        for i in range(self.row//2):
            ax = plt.subplot(1, self.row//2, i+1)
            ax.scatter(self.data[:, i], self.data[:, i+1], c=self.cent_ind)
            ax.scatter(self.cent[:, i], self.cent[:, i+1], c='r')
        plt.draw()
        plt.pause(1)

if __name__ == '__main__':
    from sklearn import datasets
    X, y = datasets.make_blobs()
    model = Kmeans(X, 3, draw=1)
    print(model.score())
    # y_pred = MiniBatchKMeans(n_clusters=2, batch_size=200, random_state=9).fit_predict(X)