import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter


class Adaboost:
    def __init__(self, n_feature=None, show_img=False):
        self.show_img = show_img

        if self.show_img:
            self.painter = Painter(n_feature)
            self.painter.beautify()
            self.painter.init_pic()

    def fit(self, data, label, max_model=10, img_save_path=None):
        data = np.array(data, dtype=float)
        label = np.array(label, dtype=float)
        n_samples = data.shape[0]
        n_features = data.shape[1]

        self.models = []

        wm = np.zeros_like(label) + 1. / n_samples

        for _ in range(max_model):
            i = np.random.randint(n_features)
            x = data[:, i]
            ems = []

            for point in x:
                # 这里没有取中位数作为划分点
                # 预先假设左边标签为1
                y = np.ones_like(label)
                y[np.where(x >= point)] = -1
                e = np.sum(wm[np.where(y != label)])

                # 左边的标签是1还是-1
                left = 1
                if e > 0.5:
                    left = -1
                    e = 1 - e

                ems.append([point, e, left])

            ems = np.array(ems)
            argmin = np.argmin(ems[:, 1])
            point, em, left = ems[argmin]

            if em == 0:
                alpha = 1
            else:
                alpha = 0.5 * np.log((1 - em) / em)

            g = np.sign((data[:, i] < point) - .1) * left

            wm *= np.exp(-alpha * label * g)
            wm /= np.sum(wm)

            self.models.append((i, alpha, point, left))

            if em < 1e-4:
                break

        if self.show_img:
            self.painter.show_pic(data, label, self.predict, img_save_path)
            self.painter.show()

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]

        pre = np.zeros(n_test)

        for a in tqdm(range(n_test)):
            for i, alpha, point, left in self.models:
                pre[a] += alpha * (left if x[a][i] < point else -left)

        return np.int_(np.sign(pre))


def sample_test():
    np.random.seed(3)
    x, y = datasets.make_blobs(centers=2, n_samples=200)
    y[y == 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = Adaboost(x.shape[1], show_img=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/Adaboost.png',
              )

    pred = model.predict(x_test)

    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 1.0"""


def real_data_test():
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    y[y == 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Adaboost()
    model.fit(x_train, y_train, max_model=30)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9649122807017544"""


def sklearn_test():
    from sklearn.ensemble import AdaBoostClassifier

    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target
    y[y == 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.956140350877193"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
    # sklearn_test()
