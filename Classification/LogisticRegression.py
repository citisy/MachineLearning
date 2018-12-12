from Regression.LinerRegression import Liner

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)


class Logistic(object):
    """
    only apply for 2 classes classification.
    """
    def __init__(self, data, label, draw=0):
        self.data = np.array(data, dtype=float)
        self.label = np.array(label)
        self.draw = draw
        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self.train()

    def train(self):
        self.model = Liner(self.data, self.label, draw=0)
        self.model.normal_equations()
        if self.draw:
            self.show()

    def predict(self, data):
        data = np.array(data, dtype=float)
        n_samples = data.shape[0]
        z = self.model.predict(data)
        sigma = 1 / (1 + np.exp(-z))
        pre = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if sigma[i] >= 0.5:
                pre[i] = 1
        return pre

    def show(self):
        fig, ax = plt.subplots()
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        x = np.arange(x_min, x_max, 0.1)
        y = np.arange(y_min, y_max, 0.1)
        x, y = np.meshgrid(x, y)
        z = self.predict(np.c_[x.ravel(), y.ravel()])
        z = z.reshape(x.shape)
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        ax.pcolormesh(x, y, z, cmap=cmap_light)
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.label)
        # fig.savefig('../img/LogisticRegression')
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    x, y = datasets.make_blobs(n_samples=200, centers=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = Logistic(x_train, y_train, draw=1)
    pre = model.predict(x_test)
    acc = np.sum(pre == y_test) / len(y_test)
    print(acc)
