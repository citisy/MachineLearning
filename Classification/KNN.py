"""
knn是lazy learning，基本不学习，网络结构很简单，但每次都有遍历所有样本计算距离，所以计算量很大。
适合大规模数据，小数据错误率高。
判定标准：“近朱者赤，近墨者黑”以及“少数服从多数”。
"""

import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)


class KNN(object):
    def __init__(self, data, label, k=10, draw=0):
        self.data = np.array(data)
        self.label = label
        self.k = k
        self.draw = draw
        self.n_sample = self.data.shape[0]
        if self.draw:
            self.show()

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]
        pre = np.zeros(n_test, dtype=int)
        for a in range(n_test):
            r = np.zeros(self.n_sample)
            for i in range(self.n_sample):
                r[i] = np.linalg.norm(x[a] - self.data[i])
            argsort = r.argsort()
            # 统计预测点附近k个点的标签，取出现次数最多的标签
            dict_ = collections.Counter(self.label[argsort[:self.k]])
            maxv = 0
            for k, v in dict_.items():
                if maxv < v:
                    pre[a] = k
                    maxv = v
        return pre

    def show(self):
        fig, ax = plt.subplots()
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        x = np.arange(x_min, x_max, 0.1)
        y = np.arange(y_min, y_max, 0.1)
        x, y = np.meshgrid(x, y)
        z = self.predict(np.c_[x.ravel(), y.ravel()])
        z = z.reshape(x.shape)
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AAAAAA', '#FFFFFF'])
        ax.pcolormesh(x, y, z, cmap=cmap_light)
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.label)
        # fig.savefig('../img/KNN.png')
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    x, y = datasets.make_blobs(centers=5, n_samples=200)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = KNN(x_train, y_train, draw=1)
    pre = model.predict(x_test)
    acc = np.sum(pre == y_test) / len(y_test)
    print(acc)
