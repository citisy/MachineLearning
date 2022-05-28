import numpy as np
from sklearn import datasets, metrics
from utils import Painter4cluster
from sklearn.cluster import KMeans


class Spectral:
    """https://arxiv.org/pdf/0711.0189.pdf
    谱聚类算法过程：
    1. 根据数据构造graph，每一节点对应一个数据点，边的权重为两点的相似度，以其邻接矩阵W表示
    2. W每一列元素相加构成一个对角矩阵，记为度矩阵D，把W-D记为拉普拉斯矩阵L
    3. 求L的前k个特征值及特征向量
    4. 前k个特征向量组成特征矩阵，用k-mean进行聚类"""

    def __init__(self, n_clusters, n_features=None, show_img=None, painter=None):
        self.n_clusters = n_clusters

        self.show_img = show_img

        if self.show_img:
            self.painter = painter or Painter4cluster(n_features)
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

    def fit_predict(self, data, gamma=1., img_save_path=None):
        n_samples = data.shape[0]

        w = np.ones((n_samples, n_samples))  # 邻接矩阵

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                w[i][j] = w[j][i] = self.cal_sim(data[i], data[j], gamma)

        d = np.diag(np.sum(w, axis=-1))  # 度矩阵
        l = d - w  # 拉普拉斯矩阵

        # 拉普拉斯矩阵标准化
        for i in range(n_samples):
            for j in range(i, n_samples):
                l[i, j] /= d[i, i] * d[j, j]
                l[j, i] = l[i, j]

        q, v = np.linalg.eig(l)  # 特征值和特征向量
        vec = v[:, np.argsort(q)[:self.n_clusters]]  # 取前k个特征向量

        pred = KMeans(n_clusters=self.n_clusters).fit_predict(vec)

        if self.show_img:
            self.painter.show_pic(data, pred, img_save_path)
            self.painter.show()

        return pred

    def cal_sim(self, x1, x2, gamma):
        """高斯核"""
        return np.exp(- gamma * np.linalg.norm(x1 - x2) ** 2)


def sample_test():
    n_clusters = 5
    x, y = datasets.make_blobs(centers=n_clusters, n_samples=500)

    model = Spectral(n_clusters=n_clusters,
                     n_features=x.shape[1], show_img=True
                     )
    pred = model.fit_predict(x,
                             # img_save_path='../img/Spectral.png',
                             )

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.980015709384567"""


def real_data_test():
    from MathMethods.Scaler import scaler

    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    x, _ = scaler.min_max(x)

    model = Spectral(n_clusters=3)
    pred = model.fit_predict(x)

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.7970657287606968"""


def sklearn_test():
    from sklearn.cluster import SpectralClustering
    from sklearn.preprocessing import MinMaxScaler

    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    x = MinMaxScaler().fit_transform(x)

    model = SpectralClustering(n_clusters=3)
    pred = model.fit_predict(x)

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.9308728982369983"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
    # sklearn_test()
