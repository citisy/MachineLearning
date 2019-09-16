from cluster import cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Spectral(cluster):
    """谱聚类算法过程：
    1. 根据数据构造graph，每一节点对应一个数据点，边的权重为两点的相似度，以其邻接矩阵W表示
    2. W每一列元素相加构成一个对角矩阵，记为度矩阵D，把W-D记为拉普拉斯矩阵L
    3. 求L的前k个特征值及特征向量
    4. 前k个特征向量组成特征矩阵，用k-mean进行聚类"""
    def train(self):
        n_samples = self.data.shape[0]
        w = np.zeros((n_samples, n_samples))    # 邻接矩阵

        for i in range(n_samples):
            for j in range(i, n_samples):
                sim = self.cal_sim(self.data[i], self.data[j])
                w[i][j] = w[j][i] = sim

        d = np.diag(np.sum(w, axis=-1))     # 度矩阵
        l = d - w   # 拉普拉斯矩阵
        q, v = np.linalg.eig(l)     # 特征值和特征向量
        vec = v[:, np.argsort(q)[:self.k]]

        pre = KMeans(n_clusters=self.k).fit_predict(vec)

        if self.draw:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=pre)
            plt.show()

    def cal_sim(self, x1, x2):
        return np.exp(-np.squeeze(np.linalg.norm(x1 - x2)))


if __name__ == '__main__':
    from sklearn.datasets import make_blobs as make_data

    np.random.seed(6)
    x, y = make_data(n_samples=500, centers=3)
    # pre = spectral_cluster(x, k=3)
    model = Spectral(x, k=3, draw=1)
