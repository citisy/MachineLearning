import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from SVM import SVM
import collections


class OvO:
    def __init__(self, base_model=SVM, n_features=None, show_img=False, show_ani=False, painter=None):
        self.base_model = base_model

        self.show_img = show_img
        self.show_ani = show_ani
        self.painter = None

        if self.show_img or self.show_ani:
            self.painter = painter or self.base_model(n_features=n_features, show_ani=show_ani).painter
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

    def fit(self, data, label, max_iter=10, img_save_path=None, ani_save_path=None):
        data = np.array(data, dtype=float)
        label = np.array(label)

        n_features = data.shape[1]

        labels = np.unique(label)
        n_labels = len(labels)

        self.models = []

        for i in range(n_labels):
            for j in range(i + 1, n_labels):
                idx = (label == labels[i]) + (label == labels[j])
                x, y = data[idx].copy(), label[idx].copy()
                pos, neg = y == i, y == j
                y[pos], y[neg] = -1, 1

                model = self.base_model(n_features=n_features, show_ani=self.show_ani, painter=self.painter)
                model.fit(x, y, max_iter=max_iter)

                self.models.append((i, j, model))

                if self.show_ani:
                    if not self.painter:
                        self.painter = model.painter

        if self.show_ani:
            self.painter.show_ani(ani_save_path)

        if self.show_img:
            self.painter.show_pic(data, label, self.predict, img_save_path)
            self.painter.show()

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]

        pres = [collections.defaultdict(int) for _ in range(n_test)]

        for a, (i, j, model) in enumerate(self.models):
            pre = model.predict(x)
            for b in range(n_test):
                if pre[b] < 0:
                    pres[b][i] += 1
                else:
                    pres[b][j] += 1

        pre = np.zeros(n_test, dtype=int)

        for i in range(n_test):
            pre[i] = max(pres[i].items(), key=lambda x: x[1])[0]

        return pre


class OvR:
    def __init__(self, base_model=SVM, n_features=None, show_img=False, show_ani=False, painter=None):
        self.base_model = base_model

        self.show_img = show_img
        self.show_ani = show_ani
        self.painter = None

        if self.show_img or self.show_ani:
            self.painter = painter or self.base_model(n_features=n_features, show_ani=show_ani).painter
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

    def fit(self, data, label, max_iter=10, img_save_path=None, ani_save_path=None):
        data = np.array(data, dtype=float)
        label = np.array(label)

        n_features = data.shape[1]

        labels = np.unique(label)
        n_labels = len(labels)

        self.models = []

        for i in range(n_labels):
            x, y = data.copy(), label.copy()
            pos, neg = y == i, y != i
            y[pos], y[neg] = 1, -1

            model = self.base_model(n_features=n_features, show_ani=self.show_ani, painter=self.painter)
            model.fit(x, y, max_iter=max_iter)

            self.models.append((i, model))

            if self.show_ani:
                if not self.painter:
                    self.painter = model.painter

        if self.show_ani:
            self.painter.show_ani(ani_save_path)

        if self.show_img:
            self.painter.show_pic(data, label, self.predict, img_save_path)
            self.painter.show()

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]

        pres = [collections.defaultdict(int) for _ in range(n_test)]

        for a, (i, model) in enumerate(self.models):
            pre = model.predict(x)

            for b in range(n_test):
                if pre[b] > 0:
                    pres[b][i] += 1

        pre = np.zeros(n_test, dtype=int)

        for i in range(n_test):  # 无人认领的点
            if not len(pres[i]):
                pres[i][len(self.models) + 1] = 0

        for i in range(n_test):
            pre[i] = max(pres[i].items(), key=lambda x: x[1])[0]

        return pre


def ovo_test():
    x, y = datasets.make_blobs(centers=3, n_samples=200)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = OvO(n_features=x.shape[1], show_img=True, show_ani=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/ovo.png',
              # ani_save_path='../img/ovo.mp4'
              )

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 1.0"""


def ovo_real_data_test():
    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    x, _, _ = Normalization().min_max(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = OvO()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9444444444444444"""


def ovr_test():
    x, y = datasets.make_blobs(centers=3, n_samples=200)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = OvR(n_features=x.shape[1], show_img=True, show_ani=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/ovr.png',
              # ani_save_path='../img/ovr.mp4'
              )

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 1.0"""


def ovr_real_data_test():
    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    x, _, _ = Normalization().min_max(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = OvR()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9722222222222222"""


if __name__ == '__main__':
    # ovo_test()
    ovo_real_data_test()
    # ovr_test()
    # ovr_real_data_test()
