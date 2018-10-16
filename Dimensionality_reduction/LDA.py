import numpy as np
import matplotlib.pyplot as plt


class LDA:
    def __init__(self, data, label, k=2):
        self.data = np.array(data)  # data -> [150, 4]
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

    data = datasets.load_iris()
    x, y = data.data, data.target
    transx = LDA(x, y).transform()
    plt.scatter(transx[:, 0], transx[:, 1], c=y)
    plt.show()
