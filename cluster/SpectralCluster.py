import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs as make_data
from sklearn.cluster import KMeans


def spectral_cluster(data, k=2):
    data = np.array(data, dtype=float)
    n_samples = data.shape[0]
    w = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            sim = cal_sim(data[i], data[j])
            w[i][j] = w[j][i] = sim

    d = np.diag(np.sum(w, axis=-1))
    l = d - w
    q, v = np.linalg.eig(l)
    vec = v[:, np.argsort(q)[:k]]

    return KMeans(n_clusters=k).fit_predict(vec)


def cal_sim(x1, x2):
    return np.exp(-np.squeeze(np.linalg.norm(x1 - x2)))


def kmeans(data, k=2):
    pass


if __name__ == '__main__':
    x, y = make_data(n_samples=500, centers=3)
    pre = spectral_cluster(x, k=3)
    plt.scatter(x[:, 0], x[:, 1], c=pre)
    plt.show()
