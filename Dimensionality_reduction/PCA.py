import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)


class PCA:
    def __init__(self, data, k=2):
        self.k = k
        self.data = np.array(data)      # data -> [150, 4]

    def transform(self):
        cov = np.cov(self.data, rowvar=False)   # cov -> [4, 4]
        eigen_values, eigen_vectors = np.linalg.eig(cov)    # eigen_values -> [4, 1], eigen_vectors -> [4, 4]

        argmax = np.argsort(eigen_values)[::-1]

        eigenvectors = eigen_vectors[:, argmax[:self.k]]    # select the first k column from eigen_vectors
                                                            # eigenvectors -> [4, 2]

        return np.matmul(self.data, eigenvectors)   # return -> [150, 2]

if __name__ == '__main__':
    from sklearn import datasets

    x, y = datasets.make_blobs(centers=3, n_features=3, n_samples=500)
    transx = PCA(x).transform()
    fig1, ax1 = plt.subplots()
    fig2, _ = plt.subplots()
    ax2 = Axes3D(fig2)
    ax1.scatter(transx[:, 0], transx[:, 1], c=y)
    ax2.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
    fig1.savefig('../img/PCA_after.png')
    fig2.savefig('../img/PCA_before.png')
    plt.show()
