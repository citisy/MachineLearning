import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MyPainter(Painter):
    def draw_GNfunc(self, dic, img_save_path=None):
        fig = plt.figure()
        for i in range(self.n_features // 2):
            a = i // self.col
            b = i % self.col

            ax3d = fig.add_subplot(projection='3d')

            ax3d.set_xlabel('x')
            ax3d.set_ylabel('y')
            ax3d.set_zlabel('z')

            for k in dic.keys():  # 作二维高斯函数的等高线和3d图像
                x = np.arange(dic[k]['mu'][i] - 3, dic[k]['mu'][i] + 3, 0.1)
                y = np.arange(dic[k]['mu'][i + 1] - 3, dic[k]['mu'][i + 1] + 3, 0.1)
                x, y = np.meshgrid(x, y)
                d = np.linalg.det(dic[k]['sigma'])
                z = np.exp(
                    -0.5 * ((x - dic[k]['mu'][0]) ** 2 + (y - dic[k]['mu'][1]) ** 2) / d ** 2) / np.sqrt(
                    2 * np.pi * d)

                # 等高线
                cs = self.ax[a][b].contour(x, y, z)
                self.ax[a][b].clabel(cs, inline=1, fontsize=10)

                ax3d.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')

        if img_save_path:
            fig.savefig(img_save_path)


class NB(object):
    def __init__(self, n_feature=None, show_img=False):
        self.show_img = show_img
        if self.show_img:
            self.painter = MyPainter(n_feature)
            self.painter.init_pic()

    def fit(self, data, label):
        data = np.array(data, dtype=float)
        label = np.array(label)

        counter = collections.Counter(label)

        return data, label, counter

    def predict(self, x):
        pass


class Gaussian(NB):
    def fit(self, data, label, img_save_path=None, img_save_path2=None):
        n_samples = data.shape[0]

        data, label, counter = super(Gaussian, self).fit(data, label)

        dic = {}
        for k, v in counter.items():  # 计算先验概率
            dic[k] = {}
            dic[k]['x'] = data[np.where(label == k)]
            dic[k]['py'] = v / n_samples

        for k in dic.keys():
            a = dic[k]['x']
            mu = np.mean(a, axis=0)
            # 无偏估计：cov = var * n / (n - 1)
            sigma = np.diagflat(np.var(a, axis=0) * len(a) / (len(a) - 1))
            dic[k]['mu'] = mu
            dic[k]['sigma'] = sigma

        self.dic = dic

        if self.show_img:
            self.painter.draw_GNfunc(self.dic, img_save_path2)
            self.painter.show_pic(data, label, self.predict, img_save_path)
            self.painter.show()

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]
        n_features = x.shape[1]

        pre = np.zeros(n_test, dtype=int)

        for a in tqdm(range(n_test)):
            cache = []
            for k in self.dic.keys():
                delta = x[a] - self.dic[k]['mu']
                sigma = self.dic[k]['sigma']
                pxy = (np.exp(-0.5 * np.matmul(np.matmul(delta.T, np.linalg.inv(sigma)), delta))
                       / np.sqrt((2 * np.pi) ** n_features * np.linalg.det(sigma)))

                cache.append((k, pxy * self.dic[k]['py']))

            pre[a] = max(cache, key=lambda x: x[1])[0]

        return pre


class Multinomial(NB):
    def fit(self, data, label, alpha=1):
        self.alpha = alpha

        n_samples = data.shape[0]
        n_features = data.shape[1]

        data, label, counter = super(Multinomial, self).fit(data, label)

        dic = {}
        for k, v in counter.items():  # 计算先验概率
            dic[k] = {}
            dic[k]['x'] = data[np.where(label == k)]
            dic[k]['ny'] = v
            dic[k]['py'] = np.log((v + self.alpha) / (n_samples + self.alpha * len(counter)))

        self.dic = dic
        self.sj = [len(np.unique(data[:, i])) for i in range(n_features)]

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]

        pre = np.zeros(n_test, dtype=int)

        for a in tqdm(range(n_test)):
            cache = []

            for k in self.dic.keys():
                pxy = 0
                ny = self.dic[k]['ny']

                for i, kx in enumerate(x[a]):
                    pxy += self.get_pxy(k, i, kx, ny)

                cache.append((k, pxy + self.dic[k]['py']))

            pre[a] = max(cache, key=lambda x: x[1])[0]

        return pre

    def get_pxy(self, k, i, kx, ny):
        nxy = np.sum(self.dic[k]['x'][:, i] == kx)
        return np.log((nxy + self.alpha) / (ny + self.sj[i] * self.alpha))  # 拉普拉斯平滑


class Bernoulli(Multinomial):
    def get_pxy(self, k, i, kx, ny):
        nxy = np.sum(self.dic[k]['x'][:, i] == 1)
        p = (nxy + self.alpha) / (ny + 2 * self.alpha)
        return np.log(p * kx + (1 - p) * (1 - kx))


def gaussian_test():
    """高斯预测"""
    x, y = datasets.make_blobs(centers=3, n_samples=200)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = Gaussian(x.shape[1], show_img=True)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 1.0"""


def real_data_gaussian_test():
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Gaussian()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9385964912280702"""


def sklearn_gaussian_test():
    from sklearn.naive_bayes import GaussianNB

    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = GaussianNB()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9385964912280702"""


def multinomial_test():
    """多项式预测"""
    x, y = datasets.make_blobs(centers=3, n_samples=200)
    x_train, x_test, y_train, y_test = train_test_split(x.astype(int), y.astype(int), test_size=0.2)

    model = Multinomial(x.shape[1], show_img=True)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 1.0"""


def real_data_multinomial_test():
    dataset = datasets.load_digits()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Multinomial()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9111111111111111"""


def sklearn_multinomial_test():
    from sklearn.naive_bayes import MultinomialNB
    dataset = datasets.load_digits()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = MultinomialNB()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9"""


def bernoulli_test():
    """伯努利判别"""
    x, y = datasets.make_blobs(centers=3, n_samples=200)

    x_train, x_test, y_train, y_test = train_test_split(np.signbit(x), np.signbit(y), test_size=0.2)

    model = Bernoulli()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 1.0"""


def real_data_bernoulli_test():
    dataset = datasets.load_digits()

    x, y = dataset.data, dataset.target
    x[x > 0] = 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Bernoulli()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.8583333333333333"""


def sklearn_bernoulli_test():
    from sklearn.naive_bayes import BernoulliNB
    dataset = datasets.load_digits()

    x, y = dataset.data, dataset.target
    x[x > 0] = 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = BernoulliNB()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.8583333333333333"""


if __name__ == '__main__':
    gaussian_test()
    # real_data_gaussian_test()
    # sklearn_gaussian_test()

    # multinomial_test()
    # real_data_multinomial_test()
    # sklearn_multinomial_test()

    # bernoulli_test()
    # real_data_bernoulli_test()
    # sklearn_bernoulli_test()
