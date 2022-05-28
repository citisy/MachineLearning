import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter


class Logistic:
    """only apply for 2 classes classification."""

    def __init__(self, n_features=None, show_img=False, show_ani=False, painter=None):
        self.show_img = show_img
        self.show_ani = show_ani

        if self.show_img or self.show_ani:
            self.painter = painter or Painter(n_features)
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

            if not painter and self.show_ani:
                self.painter.init_ani()

    def fit(self, data, label, img_save_path=None, ani_save_path=None, lr=1e-2, itera=10, batch=10):
        data = np.array(data, dtype=float)
        label = np.array(label)

        n_samples = data.shape[0]
        n_features = data.shape[1]
        self.w = np.zeros((n_features + 1,))
        self.b = .0

        data = np.pad(data, ((0, 0), (1, 0)), constant_values=1)

        for _ in tqdm(range(itera)):
            for i in range(0, n_samples, batch):
                x, y = data[i: i + batch], label[i: i + batch]

                exp = np.exp(x @ self.w)

                d = x.T @ (exp / (1 + exp)) - x.T @ y

                self.w -= lr * d / batch

                if self.show_ani:
                    self.painter.img_collections(data[:, 1:], label, self.w[1:], self.w[0])

        if self.show_ani:
            self.painter.show_ani(ani_save_path)

        if self.show_img:
            self.painter.show_pic(data[:, 1:], label, self.predict, img_save_path)
            self.painter.show()

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_samples = x.shape[0]
        x = np.pad(x, ((0, 0), (1, 0)), constant_values=1)
        z = x @ self.w
        sigma = 1 / (1 + np.exp(-z))
        pre = np.zeros(n_samples, dtype=int)
        pre[sigma >= 0.5] = 1

        return pre


def sample_test():
    x, y = datasets.make_blobs(centers=2, n_samples=200)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = Logistic(x.shape[1], show_img=True, show_ani=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/LogisticRegression.png',
              # ani_save_path='../img/LogisticRegression.mp4',
              )

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.975"""


def real_data_test():
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Logistic()
    model.fit(x_train, y_train, lr=1e-4, itera=5000)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9473684210526315"""


def sklearn_test():
    from sklearn.linear_model import LogisticRegression

    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9736842105263158"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
    # sklearn_test()
