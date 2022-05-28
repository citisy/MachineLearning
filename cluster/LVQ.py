import numpy as np
from tqdm import tqdm
from sklearn import datasets, metrics
from utils import Painter4cluster, count_distances


class MyPainter(Painter4cluster):
    def img_collections(self, data, label, centers=None, *args, **kwargs):
        for a, b, i, im in self.draw_ani(data, label, *args, **kwargs):
            im.append(self.ani_ax[a][b].scatter(centers[:, i], centers[:, i + 1], c='green',
                                                marker='*', s=100, animated=True))

    def show_pic(self, data, label, centers=None, img_save_path=None, *args, **kwargs):
        for a, b, i in self.draw_pic(data, label, img_save_path, *args, **kwargs):
            self.ax[a][b].scatter(centers[:, i], centers[:, i + 1], c='green', marker='*', s=100)


class LVQ:
    def __init__(self, n_clusters, n_features=None, show_img=None, show_ani=None, painter=None):
        self.n_clusters = n_clusters

        self.show_img = show_img
        self.show_ani = show_ani

        if self.show_img or self.show_ani:
            self.painter = painter or MyPainter(n_features)
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

            if not painter and self.show_ani:
                self.painter.init_ani()

    def fit_predict(self, data, label, lr=1e-2, max_iter=10, img_save_path=None, ani_save_path=None):
        n_samples = data.shape[0]
        n_features = data.shape[1]
        labels = np.unique(label)
        n_labels = len(labels)

        self.p = np.zeros((self.n_clusters, n_features))
        p_label = np.zeros(self.n_clusters, dtype=int)
        for i in range(self.n_clusters):
            self.p[i] = data[np.random.choice(np.where(label == labels[i % n_labels])[0])]
            p_label[i] = labels[i % n_labels]

        for _ in range(max_iter):
            p = self.p.copy()

            # 这里没有选择使用西瓜书上随机选取样本的方法，因为这样做的迭代效率太低了
            # 如果一定要使用随机选取的方法，则需要进行如下优化：
            # 每次更新完原型向量后，根据更新的幅度计算一个概率值
            # 如果更新幅度小，则下次随机到这个簇内的点的概率变低，反之则变高
            # 这是因为如果更新幅度小，意味着该簇的学习已经完毕的，所以更新的机会应该多让给其他的簇
            # j = np.random.choice(n_samples)

            for j in range(n_samples):
                i = np.argmin(count_distances(self.p, data[j], axis=1))
                if label[j] == p_label[i]:
                    self.p[i] += lr * (data[j] - self.p[i])
                else:
                    self.p[i] -= lr * (data[j] - self.p[i])

                if self.show_ani and j % 10 == 0:
                    self.painter.img_collections(data, self.predict(data), self.p)

            # 不再发生变化，退出迭代
            if np.mean(np.abs(p - self.p)) < 1e-4:
                break

        if self.show_ani:
            self.painter.show_ani(ani_save_path, fps=15)

        if self.show_img:
            self.painter.show_pic(data, self.predict(data), self.p, img_save_path)
            self.painter.show()

        return self.predict(data)

    def predict(self, x):
        n_test = x.shape[0]
        pred = np.zeros(n_test, dtype=int)

        for j in range(n_test):
            pred[j] = np.argmin(count_distances(self.p, x[j], axis=1))

        return pred


def sample_test():
    n_clusters = 5
    x, y = datasets.make_blobs(centers=n_clusters, n_samples=200)
    y_ = y.copy()
    y_[y_ == 0], y_[y_ == 1], y_[y_ == 2], y_[y_ == 3], y_[y_ == 4] = 0, 0, 1, 2, 1

    model = LVQ(n_clusters=n_clusters, n_features=x.shape[1], show_ani=True, show_img=True)
    pred = model.fit_predict(x, y_,
                             # img_save_path='../img/LVQ.png',
                             # ani_save_path='../img/LVQ.mp4'
                             )
    print(pred)
    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 1.0"""


if __name__ == '__main__':
    sample_test()
