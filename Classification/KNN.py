import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.colors import ListedColormap
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)


class KNN:
    """
    knn是lazy learning，基本不学习，网络结构很简单，但每次都有遍历所有样本计算距离，所以计算量很大。
    适合大规模数据，小数据错误率高。
    判定标准：“近朱者赤，近墨者黑”以及“少数服从多数”。
    """
    def __init__(self, data, label, k=10, show_img=False, img_save_path=None):
        self.data = np.array(data, dtype=float)
        self.label = label
        self.k = k
        self.show_img = show_img
        self.img_save_path = img_save_path
        self.n_sample = self.data.shape[0]

        if self.show_img:
            self.show()

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]
        pre = np.zeros(n_test, dtype=int)

        for a in range(n_test):
            r = np.zeros(self.n_sample)

            # 计算预测点到每个样本的距离
            for i in range(self.n_sample):
                r[i] = np.linalg.norm(x[a] - self.data[i])

            # 统计预测点附近k个点的标签，取出现次数最多的标签
            argsort = r.argsort()
            pre[a] = collections.Counter(self.label[argsort[:self.k]]).most_common(1)[0][0]

        return pre

    def show(self):
        fig, ax = plt.subplots()

        # 设置作图范围
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1

        xx = np.linspace(x_min, x_max, 200)
        yy = np.linspace(y_min, y_max, 200)
        xv, yv = np.meshgrid(xx, yy)
        zv = self.predict(np.c_[xv.ravel(), yv.ravel()])
        zv = zv.reshape(xv.shape)

        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AAAAAA', '#FFFFFF'])
        ax.pcolormesh(xv, yv, zv, cmap=cmap_light)
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.label)

        if not self.img_save_path:
            fig.savefig(img_save_path)

        plt.show()


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    np.random.seed(6)

    x, y = datasets.make_blobs(centers=5, n_samples=200)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # img_save_path = None
    img_save_path = '../img/KNN.png'
    model = KNN(x_train, y_train, show_img=True, img_save_path=img_save_path)
    pre = model.predict(x_test)
    acc = np.sum(pre == y_test) / len(y_test)
    print('accurate: ', acc)
