"""
SOM: Self-organizing Maps
适合环装数据的聚类
https://zhuanlan.zhihu.com/p/31637590
https://wenku.baidu.com/view/74927ae8aeaad1f346933f42.html?qq-pf-to=pcqq.c2c
https://blog.csdn.net/wj176623/article/details/52526617
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)


class SOM(object):
    def __init__(self, data, lr=1, itera=10, batch_size=10, output_size=2, show_img=False):
        """
        :param data: input data
                    size -> [n_sample, input_size], input_size -> num of features
        :param lr: learning rate, will be change during training
        :param itera: max iteration
        :param batch_size: num of data when training
        :param output_size: num of classes
        """
        self.raw_data = np.array(data, dtype=float)
        self.data = np.array(data, dtype=float)
        self.lr = lr
        self.itera = itera
        self.batch_size = batch_size
        self.output_size = output_size
        self.show_img = show_img
        self.n_sample = self.data.shape[0]
        self.intput_size = self.data.shape[1]
        self.w = np.random.rand(self.output_size, self.intput_size)
        self.output = np.zeros(self.n_sample)

        if self.show_img:
            self.col = math.ceil(np.sqrt(self.intput_size / 2))
            self.row = math.ceil(self.intput_size / 2 / self.col)

            self.ims = []
            self.fig, self.ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)
            self.fig.set_tight_layout(True)

            self.ims2 = []
            self.fig2, self.ax2 = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)
            self.fig2.set_tight_layout(True)

        self.norm(self.data)
        self.train()

    def norm(self, data):
        """
        data is normalized by unit circle
        """
        for i in range(len(data)):
            data[i] /= np.linalg.norm(data[i])

        return data

    def train(self, **kwargs):
        t = 0
        for i in range(self.itera):
            self.img_collections(self.data, self.w)

            n = self.getn(t)
            for j in range(self.n_sample):
                self.w = self.norm(self.w)
                self.d = np.matmul(self.w, self.data[j].T)
                argmax = np.argmax(self.d)
                neighbor = self.get_neighbor(argmax, n)
                for k, v in neighbor.items():
                    self.w[k] += self.update_lr(self.lr, t, v) * (self.data[j] - self.w[k])

            t += 1

        self.show_gif(kwargs.get('img1_save_path'), kwargs.get('img2_save_path'))

    def update_lr(self, lr, t, n):
        return lr * np.exp(-(n * self.output_size)) / (t + 1)

    def getn(self, t):
        return 1 - t / self.itera

    def get_neighbor(self, i, n):
        neighbor = {}
        for a in range(self.output_size):
            r = np.linalg.norm(self.w[a] - self.w[i])
            if r < n:
                neighbor[a] = r
        return neighbor

    def predict(self, data, w):
        test_data = np.array(data)
        test_data = self.norm(test_data)
        d = np.matmul(w, test_data.T)
        point_index = np.argmax(d, axis=0)
        return point_index

    def img_collections(self, data, w):
        if self.show_img:
            point_index = self.predict(data, w)
            im = []
            im2 = []
            for i in range(self.intput_size // 2):
                a = i // self.col
                b = i % self.col
                im.append(self.ax[a][b].scatter(data[:, i], data[:, i + 1], c=point_index, animated=True))
                im2.append(self.ax2[a][b].scatter(self.raw_data[:, i], self.raw_data[:, i + 1],
                                                  c=point_index, animated=True))

            self.ims.append(im)
            self.ims2.append(im2)

    def show_gif(self, img1_save_path, img2_save_path):
        if self.show_img:
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=2000 / len(self.ims), blit=True,
                                            repeat_delay=0, repeat=True)
            ani2 = animation.ArtistAnimation(self.fig2, self.ims2, interval=2000 / len(self.ims2), blit=True,
                                             repeat_delay=0, repeat=True)
            if img1_save_path is not None:
                ani.save(img1_save_path, writer='imagemagick')

            if img2_save_path is not None:
                ani2.save(img2_save_path, writer='imagemagick')

            plt.show()


if __name__ == '__main__':
    from sklearn import datasets

    np.random.seed(7)
    x, y = datasets.make_blobs(n_samples=500, centers=4, n_features=2)
    model = SOM(x, output_size=4, itera=20, show_img=True)
    model.train()
    # model.train(img1_save_path='../img/SOM_before_train.gif', img2_save_path='../img/SOM_after_train.gif')

    # the num of prediction classes mill be less than output_size
    # pre = model.predict(model.data)
    # model.show()
