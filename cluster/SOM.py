import numpy as np
from sklearn import datasets, metrics
from utils import Painter4cluster
from MathMethods.Scaler import scaler


class MyPainter(Painter4cluster):
    def img_collections(self, data, label, *args, **kwargs):
        for a, b, i, im in self.draw_ani(data, label, *args, **kwargs):
            x = scaler.vec(data, axis=1)
            im.append(self.ani_ax[a][b].scatter(x[:, i], x[:, i + 1], c=label, alpha=0.5, animated=True))


class SOM:
    def __init__(self, n_clusters, batch_size=10, n_features=None, show_img=None, show_ani=None, painter=None):
        """
        :param batch_size: num of data when training
        :param n_clusters: num of classes
        """

        self.batch_size = batch_size
        self.output_size = n_clusters

        self.show_img = show_img
        self.show_ani = show_ani

        if self.show_img or self.show_ani:
            self.painter = painter or MyPainter(n_features)
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

            if not painter and self.show_ani:
                self.painter.init_ani()

    def fit_predict(self, data, lr=0.5, max_iter=10, img_save_path=None, ani_save_path=None):
        """
        :param data: input data
                    size -> [n_sample, input_size], input_size -> num of features
        :param lr: learning rate, will be change during training
        :return:
        """
        x = scaler.vec(data, axis=1)
        n_samples = x.shape[0]
        self.w = data[np.random.choice(n_samples, self.output_size, replace=False)]

        for i in range(max_iter):
            # 计算邻域半径
            r = 1 - i / max_iter

            for j in np.random.choice(n_samples, self.batch_size, replace=False):
                # 获胜神经元
                w = np.argmax(np.matmul(self.w, x[j].T))

                # 获取邻域半径内的样本点
                neighbor = self.get_neighbor(w, r)

                for k, r_ in neighbor.items():
                    lr_ = lr / (i + 1) * np.exp(-(r * self.output_size))
                    self.w[k] += lr_ * (x[j] - self.w[k])
                    self.w[k] = scaler.vec(self.w[k])

                    if self.show_ani:
                        self.painter.img_collections(data, self.predict(data))

        if self.show_ani:
            self.painter.show_ani(ani_save_path)

        if self.show_img:
            self.painter.show_pic(data, self.predict(data), img_save_path)
            self.painter.show()

        return self.predict(x)

    def get_neighbor(self, i, n):
        neighbor = {}
        for a in range(self.output_size):
            r = np.linalg.norm(self.w[a] - self.w[i])
            if r < n:
                neighbor[a] = r
        return neighbor

    def predict(self, x):
        x = scaler.vec(x, axis=1)
        pred = np.argmax(np.matmul(self.w, x.T), axis=0)
        return pred


def sample_test():
    np.random.seed(21)
    n_clusters = 4
    x, y = datasets.make_blobs(centers=n_clusters, n_samples=500)

    x, _ = scaler.mean(x)

    model = SOM(n_clusters=n_clusters,
                n_features=x.shape[1], show_ani=True, show_img=True
                )
    pred = model.fit_predict(x,
                             # img_save_path='../img/SOM.png',
                             # ani_save_path='../img/SOM.mp4'
                             )

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 1.0"""


def real_data_test():
    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    x, _ = scaler.mean(x)

    model = SOM(n_clusters=3)
    pred = model.fit_predict(x)

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.76759989942771"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
