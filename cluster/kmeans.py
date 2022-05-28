"""
　　(1) 任意选择k个对象作为初始的簇中心；
　　(2) repeat；
　　(3) 根据簇中对象的平均值，将每个对象(重新)赋予最类似的簇；
　　(4) 更新簇的平均值，即计算每个簇中对象的平均值；
　　(5) until不再发生变化。
"""
import numpy as np
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


class Kmeans:
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

    # 模型训练
    def fit_predict(self, data, itera=100, img_save_path=None, ani_save_path=None):
        n_samples = data.shape[0]
        n_features = data.shape[1]

        self.centers = data[np.random.choice(n_samples, self.n_clusters, False)]

        for _ in range(itera):
            pred = self.predict(data)

            new_centers = np.zeros((self.n_clusters, n_features))

            for k in range(self.n_clusters):
                idx = pred == k
                if np.any(idx):
                    new_centers[k] = np.mean(data[idx], axis=0)
                else:
                    new_centers[k] = data[np.random.randint(n_samples)]

            # 中心点位置不变，训练完成，退出循环
            if np.mean(np.abs(self.centers - new_centers)) < 1e-4:
                break

            if self.show_ani:
                self.painter.img_collections(data, pred, self.centers)

            self.centers = new_centers

        if self.show_ani:
            self.painter.show_ani(ani_save_path, fps=5)

        if self.show_img:
            self.painter.show_pic(data, self.predict(data), self.centers, img_save_path)
            self.painter.show()

        return self.predict(data)

    def predict(self, x):
        n_samples = x.shape[0]
        pred = np.zeros(n_samples, dtype=int)

        for a in range(n_samples):
            # 求每个点与每个中心点的距离
            r = count_distances(x[a], self.centers, axis=1)

            # 记录离点最近的中心点的下标
            pred[a] = np.argmin(r)

        return pred


def sample_test():
    np.random.seed(3)
    n_clusters = 5
    x, y = datasets.make_blobs(centers=n_clusters, n_samples=200)

    model = Kmeans(n_clusters=n_clusters, n_features=x.shape[1], show_ani=True, show_img=True)
    pred = model.fit_predict(x,
                             # img_save_path='../img/Kmeans.png',
                             # ani_save_path='../img/Kmeans.mp4'
                             )

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.9749679718552928"""


def real_data_test():
    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    model = Kmeans(n_clusters=3)
    pred = model.fit_predict(x)

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.37111371823084754"""


def sklearn_test():
    from sklearn.cluster import KMeans

    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    model = KMeans(n_clusters=3)
    model.fit(x)
    pred = model.predict(x)

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.37111371823084754"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
    # sklearn_test()
