import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter
from sklearn.tree import DecisionTreeClassifier


class RF:
    """这里没有加入随机抽取特征进行分裂，所以准确来说这只是一个bagging"""

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
        self.models = []

        for _ in tqdm(range(max_model)):
            idx = np.random.choice(n_samples, n_samples)
            x, y = data[idx], label[idx]
            model = DecisionTreeClassifier()
            model.fit(x, y)
            self.models.append(model)

        if self.show_img:
            self.painter.show_pic(data, label, self.predict, img_save_path)
            self.painter.show()

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]

        pres = np.zeros((n_test, len(self.models)), dtype=int)
        pre = np.zeros(n_test, dtype=int)

        for i, model in enumerate(self.models):
            pres[:, i] = model.predict(x)

        for i, arr in enumerate(pres):
            pre[i] = np.argmax(np.bincount(arr))

        return pre


def sample_test():
    x, y = datasets.make_blobs(centers=5, n_samples=200)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = RF(x.shape[1], show_img=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/RandomForest.png',
              )

    pred = model.predict(x_test)

    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 1.0"""


def real_data_test():
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = RF()
    model.fit(x_train, y_train, max_model=30)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.956140350877193"""


def sklearn_test():
    from sklearn.ensemble import RandomForestClassifier

    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9649122807017544"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
    # sklearn_test()
