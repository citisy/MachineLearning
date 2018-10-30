# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)


class Liner(object):
    def __init__(self, data, label, draw=0):
        self.raw_data = np.array(data, dtype=float)
        self.data = np.array(data, dtype=float)
        self.label = np.array(label)
        self.label = np.reshape(self.label, (-1, 1))
        self.draw = draw
        self.n_samples = np.shape(data)[0]
        self.n_features = np.shape(data)[1]
        self.pole = []
        for i in range(self.n_features):
            self.pole.append(max(abs(self.data[:, i].min()), abs(self.data[:, i].max())))

        self.data = self.norm(self.data)
        """
        before concatenate, x = [x1, x2, ...]
        >>>data.shape
        >>>(100, 1)

        after concatenate, x = [x0, x1, x2, ...]:
        >>>data.shape
        >>>(100, 2)

        in such example, data is x axis, label is y axis
        """
        self.data = np.concatenate((np.ones((self.n_samples, 1)), self.data), axis=1)
        self.raw_data = np.concatenate((np.ones((self.n_samples, 1)), self.data), axis=1)

        if self.draw:
            self.ims = []
            self.col = math.ceil(np.sqrt(self.n_features/2))
            self.row = math.ceil(self.n_features/2/self.col)
            self.fig, self.ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)
            self.ax[0][0].set_ylim(self.label.min()*1.2, self.label.max()*1.2)
            self.fig.set_tight_layout(True)

    def norm(self, data):
        """
        before norm:
        >> data
        >>[[-3.62194721 -5.49173113]
         [-3.23435367 -4.67226512]
         [-1.58990744 -9.87007247]
         [ 1.95358937 -1.92006285]
         [ 2.68055418 -1.53837307]]
        after norm
        >>data
        >>[[-0.83333333 -0.46366859]
         [-0.74415627 -0.39448082]
         [-0.36580402 -0.83333333]
         [ 0.44947953 -0.16211151]
         [ 0.61673874 -0.12988532]]
        all data will fall between [-1, 1]
        """
        for i in range(self.n_features):
            data[:, i] /= self.pole[i] * 1.2

        return data

    def gradient_descent(self, lr=1e-2, itera=100):
        """
        :param lr: learning rate, if too big, the model can't be convergence
        :param itera: iteration, it's big while lr is small
        :return:
        """
        xmean = np.mean(self.data, axis=0)
        ymean = np.mean(self.label)
        self.w = np.mat(ymean / xmean).T
        xMat = np.mat(self.data)
        yMat = np.mat(self.label)
        for i in range(itera):
            if self.draw:
                self.show(self.data, self.label)
            self.w += lr * (xMat.T * (yMat - xMat * self.w))

        if self.draw:
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=1000 / len(self.ims), blit=True,
                                            repeat_delay=1000, repeat=False)
            # ani.save('../img/LinerRegression.gif', writer='pillow', fps=1000)
            plt.show()

    def normal_equations(self):
        """
        w = inv(x'x)(x'y)
        """
        xMat = np.mat(self.data)
        yMat = np.mat(self.label)
        xTx = xMat.T * xMat
        self.w = xTx.I * (xMat.T * yMat)
        if self.draw:
            self.show(self.data, self.label)
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=1, blit=True,
                                            repeat_delay=500, repeat=False)
            # ani.save('img/dbscan.gif', writer='pillow', fps=1000)
            plt.show()

    def predict(self, data):
        data = np.array(data)
        data = np.concatenate((np.ones((len(data), 1)), data), axis=1)
        pre = data * self.w
        return pre

    def show(self, data, label):
        im = []
        x = np.linspace(data[:, 1].min(), data[:, 1].max(), 5).reshape(-1, 1)
        y = self.predict(x)
        sca = self.ax[0][0].scatter(data[:, 1], label, c='b', animated=True)
        line, = self.ax[0][0].plot(x, y, c='r', animated=True)
        im.append(sca)
        im.append(line)
        self.ims.append(im)


if __name__ == '__main__':
    from sklearn import datasets

    x, y = datasets.make_regression(n_samples=100, n_features=1, random_state=0, noise=4.0,
                                    bias=100.0)
    model = Liner(x, y, draw=1)
    model.gradient_descent()

    pre = model.predict(model.norm(x))
    print(pre)
    print(y)

