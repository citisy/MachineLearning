import numpy as np
from sklearn import datasets
from utils import Painter4decomposition


class PCA:
    def __init__(self, k=2, show_img=False):
        self.k = k

        self.show_img = show_img

        if self.show_img:
            self.painter = Painter4decomposition()
            self.painter.beautify()
            self.painter.init_pic()

    def fit_transform(self, x, y=None, img_save_path=None):
        x = np.array(x)

        cov = np.cov(x, rowvar=False)

        eigen_values, eigen_vectors = np.linalg.eig(cov)

        eigenvectors = eigen_vectors[:, np.argsort(eigen_values)[::-1][:self.k]]

        x_ = np.matmul(x, eigenvectors)

        if self.show_img:
            self.painter.show_pic(x, y, x_, img_save_path)
            self.painter.show()

        return x_


def sample_test():
    x, y = datasets.make_s_curve(n_samples=500)
    # x, y = datasets.make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3,
    #                                     n_informative=2, n_clusters_per_class=1, class_sep=0.5)

    model = PCA(show_img=True)
    x_ = model.fit_transform(x, y,
                             # img_save_path='../img/PCA.png',
                             # img_save_path='../img/PCA2.png'
                             )


def sklearn_test():
    from sklearn.decomposition import PCA

    x, y = datasets.make_s_curve(n_samples=500)

    model = PCA()
    x_ = model.fit_transform(x)

    painter = Painter4decomposition()
    painter.beautify()
    painter.init_pic()
    painter.show_pic(x, y, x_)
    painter.show()


if __name__ == '__main__':
    sample_test()
    # sklearn_test()
