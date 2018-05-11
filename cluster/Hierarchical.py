#-*- coding: utf-8 -*-

'''
　(1) 将每个对象看作一类，计算两两之间的最小距离；
　(2) 将距离最小的两个类合并成一个新类；
　(3) 重新计算新类与所有类之间的距离；
　(4) 重复(2)、(3)，直到所有类最后合并成一类
'''


import numpy as np
import matplotlib.pyplot as plt
from cluster import cluster

class Hierarchical(cluster):
    # TODO 耗时长
    def train(self):
        self.cent_ind = np.array(range(self.col))
        for _ in range(self.col - self.k):
            r = np.zeros((self.col, self.col))
            r += 10
            for i in range(self.col):
                for j in range(i+1, self.col):
                    if self.cent_ind[i] == self.cent_ind[j]:
                        r[i][j] = 10
                    else:
                        r[i][j] = self.get_r(self.data[i], self.data[j])
            min_ind = np.argmin(r)
            j = min_ind % self.col
            i = min_ind // self.col
            # 距离最近的簇只保留一个
            ind = np.where(self.cent_ind == self.cent_ind[max(i, j)])
            self.cent_ind[ind] = self.cent_ind[min(i, j)]
            if self.draw:
                self.show_()
        print('train completed!')
        plt.show()

    # 二维可视化
    def show_(self):
        plt.clf()
        for i in range(self.row//2):
            ax = plt.subplot(1, self.row//2, i+1)
            ax.scatter(self.data[:, i], self.data[:, i+1], c=self.cent_ind)
        plt.draw()
        plt.pause(0.01)



if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.cluster import AgglomerativeClustering
    X, y = datasets.make_moons(n_samples=100,noise=0.1)
    model = Hierarchical(X, 2, draw=0)
    # model = AgglomerativeClustering(n_clusters=2).fit(X)


