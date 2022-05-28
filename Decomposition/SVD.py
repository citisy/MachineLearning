import numpy as np
from sklearn import datasets
from utils import Painter4decomposition


class SVD:
    def __init__(self, k=2, show_img=False):
        self.k = k

        self.show_img = show_img

        if self.show_img:
            self.painter = Painter4decomposition()
            self.painter.beautify()
            self.painter.init_pic()

    def fit_transform(self, x, y=None, img_save_path=None):
        """x_ = x dot V_k^T"""
        x = np.array(x)

        s, v = np.linalg.eig(x.T @ x)

        x_ = x @ v[:, np.argsort(s)[::-1][:self.k]]

        if self.show_img:
            self.painter.show_pic(x, y, x_, img_save_path)
            self.painter.show()

        return x_


def sample_test():
    x, y = datasets.make_s_curve(n_samples=500)

    model = SVD(show_img=True)
    x_ = model.fit_transform(x, y,
                             # img_save_path='../img/SVD.png'
                             )


def sklearn_test():
    from sklearn.decomposition import TruncatedSVD

    x, y = datasets.make_s_curve(n_samples=500)

    model = TruncatedSVD(n_components=2)
    x_ = model.fit_transform(x)

    painter = Painter4decomposition()
    painter.beautify()
    painter.init_pic()
    painter.show_pic(x, y, x_)
    painter.show()


if __name__ == '__main__':
    sample_test()
    # sklearn_test()
