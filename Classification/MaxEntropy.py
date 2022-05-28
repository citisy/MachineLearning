import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter
import collections


class MaxEnt:
    def __init__(self, n_features=None, show_img=False):
        self.show_img = show_img

        if self.show_img:
            self.painter = Painter(n_features)
            self.painter.beautify()
            self.painter.init_pic()

    def fit(self, data, label, itera=10):
        data = np.array(data, dtype=float)
        label = np.array(label)

        n_samples = data.shape[0]
        n_features = data.shape[1]
        self.pxy = collections.defaultdict(float)
        self.px = collections.defaultdict(float)
        w = np.zeros(n_features)

        for i in range(n_samples):
            for j in range(n_features):
                x, y = data[i, j], label[i]
                self.px[(x, j)] += 1 / n_samples
                self.pxy[(x, y, j)] += 1 / n_samples

        counter = collections.Counter(label)

        fs = np.zeros_like(data)

        for i in range(n_samples):
            for j in range(n_features):
                x, y = data[i, j], label[j]
                fs[i, j] = self.f(x, y, j)

        for _ in tqdm(range(itera)):
            pws = np.exp(fs * w)
            pws = pws / np.sum(pws, axis=1).reshape((-1, 1)).repeat(n_features, axis=1)

            for j in range(n_features):
                ep, ep_ = 0, 0
                for i in range(n_samples):
                    x, y = data[i, j], label[i]
                    px = self.px[(x, j)]
                    pxy = self.pxy[(x, y, j)]
                    pw = pws[i, j]
                    f = fs[i, j]

                    ep += px * pw * f
                    ep_ += pxy * f

                m = np.sum(fs[:, j])
                if m:
                    delta = 1 / m * np.log(ep_ / ep)
                    w[j] += delta

        self.w = w
        self.counter = counter

        if self.show_img:
            self.painter.show_pic(data, label, self.predict)
            self.painter.show()

    def f(self, x, y, j):
        return x + 1 if (x, y, j) in self.pxy else 0

    def predict(self, x):
        data = np.array(x, dtype=float)
        n_test = data.shape[0]
        n_features = data.shape[1]

        pre = np.zeros(n_test, dtype=int)

        for i in tqdm(range(n_test)):
            h = {k: 0 for k in self.counter}
            for j in range(n_features):
                x = data[i, j]

                for k in self.counter:
                    f, w = self.f(x, k, j), self.w[j]
                    if f and w:
                        pw = w * f
                        h[k] -= pw * np.log2(pw)  # 由于最后进行argmax操作，所以这里忽略了归一化因子

            pre[i] = max(h.items(), key=lambda x: x[1])[0]

        return pre


def sample_test():
    dataset = datasets.load_digits()
    x, y = dataset.data, dataset.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = MaxEnt()

    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.6805555555555556"""


if __name__ == '__main__':
    sample_test()
