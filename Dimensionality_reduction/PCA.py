import numpy as np
import matplotlib.pyplot as plt


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
    data = datasets.load_iris()
    x, y = data.data, data.target
    transx = PCA(x).transform()
    plt.scatter(transx[:, 0], transx[:, 1], c=y)
    plt.show()
