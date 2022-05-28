import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter


class MyPainter(Painter):
    def show_pic(self, data, label, w, predict, img_save_path=None, *args, **kwargs):
        self.cmap_light = kwargs.get('cmap_light', None)
        for a, b, i in self.draw_pic(data, label, predict, img_save_path, *args, **kwargs):
            x_min, x_max = data[:, i].min() - 1, data[:, i].max() + 1
            xx = np.arange(x_min, x_max)
            y = -(w[1] * xx + w[0]) / w[2]
            self.ax[a][b].plot(xx, y, c='black')


class Linear(object):
    def __init__(self, n_features=None, show_img=False, show_ani=False, painter=None, fit_method='ne'):
        self.show_img = show_img
        self.fit_method = fit_method
        self.show_ani = show_ani

        if self.show_img or self.show_ani:
            self.painter = painter or MyPainter(n_features)
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

            if not painter and self.show_ani and self.fit_method == 'gd':
                self.painter.init_ani()

    def fit(self, data, label, img_save_path=None, ani_save_path=None, lr=1e-2, itera=10, **kwargs):
        data = np.array(data, dtype=float)
        label = np.array(label)

        if self.fit_method == 'gd':
            self.gradient_descent(data, label, lr, itera)

        else:
            self.normal_equations(data, label)

        if self.show_ani and self.fit_method == 'gd':
            self.painter.show_ani(ani_save_path)

        if self.show_img:
            self.painter.show_pic(data, label, self.w, self.predict, img_save_path, **kwargs)
            self.painter.show()

    def gradient_descent(self, data, label, lr=1e-2, itera=100):
        data = np.pad(data, ((0, 0), (1, 0)), constant_values=1)

        self.w = (np.mean(label) / np.mean(data, axis=0)).T

        for _ in tqdm(range(itera)):
            if self.show_ani:
                self.painter.img_collections(data[:, 1:], label, self.w[1:], self.w[0])

            self.w += lr * (data.T @ (label - data @ self.w))

    def normal_equations(self, data, label):
        """w = inv(x'x)(x'y)"""
        data = np.pad(data, ((0, 0), (1, 0)), constant_values=1)
        self.w = np.linalg.inv(data.T @ data) @ data.T @ label

    def predict(self, x):
        x = np.array(x, dtype=float)
        x = np.pad(x, ((0, 0), (1, 0)), constant_values=1)
        pre = x @ self.w
        return pre


def simple_test():
    x, y = datasets.make_regression(n_samples=200, n_features=2, noise=20.0,
                                    random_state=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = Linear(x.shape[1], show_img=True, fit_method='gd')
    model.fit(x_train, y_train,
              # ani_save_path='../img/LinearRegression.mp4',
              # img_save_path='../img/LinearRegression.png',
              )

    pred = model.predict(x_test)
    print("loss:", np.linalg.norm(pred - y_test))
    """loss: 153.53852106142685"""


def real_data_test():
    dataset = datasets.load_diabetes()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Linear(fit_method='ne')
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print("loss:", np.linalg.norm(pred - y_test))
    """loss: 513.1990899257864"""


def sklearn_test():
    from sklearn.linear_model import LinearRegression

    dataset = datasets.load_diabetes()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LinearRegression()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print("loss:", np.linalg.norm(pred - y_test))
    """loss: 513.1990899257862"""


if __name__ == '__main__':
    simple_test()
    # real_data_test()
    # sklearn_test()
