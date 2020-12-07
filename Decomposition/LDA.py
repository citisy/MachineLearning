import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# sns.set(style="white", palette="muted", color_codes=True)


class LDA:
    def __init__(self, data, label, k=2, draw=0):
        self.data = np.array(data, dtype=float)  # data -> [150, 4]
        self.label = np.array(label)  # label -> [150, 1]
        self.k = k
        self.cl = np.unique(label)  # cl: values -> [0, 1, 2]

    def transform(self):
        means = np.zeros((len(self.cl), np.shape(self.data)[1]))  # means -> [3, 4]
        for i, c in enumerate(self.cl):
            means[i] = np.mean(self.data[self.label == c], axis=0)
        mean = np.mean(means, axis=0)  # mean -> [3, 1]
        sw = np.zeros((np.shape(self.data)[1], np.shape(self.data)[1]))  # sw -> [4, 4]
        sb = np.zeros((np.shape(self.data)[1], np.shape(self.data)[1]))  # sb -> [4, 4]
        for i, c in enumerate(self.cl):
            submean = means[i] - mean[i]  # submean -> [1, 4]
            subx = self.data[self.label == c] - means[i]  # subx -> [?, 4]
            sb += np.sum(self.label == c) * np.matmul(submean.T, submean)
            sw += np.matmul(subx.T, subx)

        s = np.matmul(np.linalg.inv(sw), sb)
        eigen_values, eigen_vectors = np.linalg.eig(s)  # eigen_values -> [4, 1], eigen_vectors -> [4, 4]

        argmax = np.argsort(eigen_values)[::-1]

        eigenvectors = eigen_vectors[:, argmax[:self.k]]  # select the first k column from eigen_vectors
                                                          # eigenvectors -> [4, 2]

        return np.matmul(self.data, eigenvectors)  # return -> [150, 2]


if __name__ == '__main__':
    from sklearn import datasets

    # x, y = datasets.make_blobs(centers=3, n_features=3, n_samples=500)
    x, y = datasets.make_s_curve(n_samples=1000)

    # transx = LDA(x, y, draw=1).transform()
    # fig1, ax1 = plt.subplots()
    fig2, _ = plt.subplots()
    ax2 = Axes3D(fig2)
    # ax1.scatter(transx[:, 0], transx[:, 1], c=y)
    ax2.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=plt.cm.Spectral)
    ax2.view_init(4, -72)
    # fig1.savefig('../img/LDA_after.png')
    # fig2.savefig('../img/LDA_before.png')
    plt.show()
