from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)


class Spectral:
    """谱聚类算法过程：
    1. 根据数据构造graph，每一节点对应一个数据点，边的权重为两点的相似度，以其邻接矩阵W表示
    2. W每一列元素相加构成一个对角矩阵，记为度矩阵D，把W-D记为拉普拉斯矩阵L
    3. 求L的前k个特征值及特征向量
    4. 前k个特征向量组成特征矩阵，用k-mean进行聚类"""

    def __init__(self, data, k=3, show_img=False):
        self.data = np.array(data, dtype=float)
        self.k = k  # 簇数
        self.show_img = show_img
        self.n_features = self.data.shape[1]  # 数据维度
        self.n_samples = self.data.shape[0]  # 数据数量
        self.point_index = np.array(range(self.n_samples), dtype=int)  # 每一个点的归属簇

    def train(self):
        w = np.zeros((self.n_samples, self.n_samples))  # 邻接矩阵

        for i in range(self.n_samples):
            for j in range(i, self.n_samples):
                sim = self.cal_sim(self.data[i], self.data[j])
                w[i][j] = w[j][i] = sim

        d = np.diag(np.sum(w, axis=-1))  # 度矩阵
        l = d - w  # 拉普拉斯矩阵
        q, v = np.linalg.eig(l)  # 特征值和特征向量
        vec = v[:, np.argsort(q)[:self.k]]

        pre = KMeans(n_clusters=self.k).fit_predict(vec)

        if self.show_img:
            plt.scatter(self.data[:, 0], self.data[:, 1], c=pre)
            plt.show()

    def cal_sim(self, x1, x2):
        return np.exp(-np.squeeze(np.linalg.norm(x1 - x2)))


if __name__ == '__main__':
    from sklearn.datasets import make_blobs as make_data

    np.random.seed(6)
    x, y = make_data(n_samples=500, centers=3)

    model = Spectral(x, k=3, show_img=True)
    model.train()
