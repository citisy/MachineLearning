import numpy as np
from sklearn import datasets
from utils import Painter4decomposition


class LDA:
    def __init__(self, k=2, show_img=False):
        self.k = k

        self.show_img = show_img

        if self.show_img:
            self.painter = Painter4decomposition()
            self.painter.beautify()
            self.painter.init_pic()

    def fit_transform(self, x, y=None, img_save_path=None):
        x = np.array(x)
        n_features = x.shape[1]
        cl = np.unique(y)

        means = np.zeros((len(cl), n_features))

        for i, c in enumerate(cl):
            means[i] = np.mean(x[y == c], axis=0)

        mean = np.mean(means, axis=0)

        sw = np.zeros((n_features, n_features))
        sb = np.zeros((n_features, n_features))

        for i, c in enumerate(cl):
            submean = means[i] - mean[i]
            sb += np.sum(y == c) * np.matmul(submean.T, submean)

            subx = x[y == c] - means[i]
            sw += np.matmul(subx.T, subx)

        s = np.matmul(np.linalg.inv(sw), sb)
        eigen_values, eigen_vectors = np.linalg.eig(s)

        eigenvectors = eigen_vectors[:, np.argsort(eigen_values)[::-1][:self.k]]

        x_ = np.matmul(x, eigenvectors)

        if self.show_img:
            self.painter.show_pic(x, y, x_, img_save_path)
            self.painter.show()

        return x_


def sample_test():
    x, y = datasets.make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3,
                                        n_informative=2, n_clusters_per_class=1, class_sep=0.5)

    model = LDA(show_img=True)
    x_ = model.fit_transform(x, y,
                             # img_save_path='../img/LDA.png'
                             )


def sklearn_test():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    x, y = datasets.make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3,
                                        n_informative=2, n_clusters_per_class=1, class_sep=0.5)

    model = LinearDiscriminantAnalysis(n_components=2)
    x_ = model.fit_transform(x, y)

    painter = Painter4decomposition()
    painter.beautify()
    painter.init_pic()
    painter.show_pic(x, y, x_)
    painter.show()


if __name__ == '__main__':
    sample_test()
    # sklearn_test()
