import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import time

np.random.seed(6)


class Painter:
    cmap_light = ListedColormap(['#AAFFAA', '#AAAAFF', '#FFFFAA', '#AAAAAA', '#FFFFFF', '#FFAAAA'])

    def __init__(self, n_features):
        self.n_features = n_features
        self.col = int(np.ceil(np.sqrt(self.n_features / 2)))
        self.row = int(np.ceil(self.n_features / 2 / self.col))

    def beautify(self):
        import seaborn as sns
        sns.set(style="white", palette="muted", color_codes=True)

    def init_pic(self):
        self.fig, self.ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)

    def draw_pic(self, data, label, predict, img_save_path=None, *args, **kwargs):
        for i in range(0, self.n_features, 2):
            a = i // self.col
            b = i % self.col
            yield a, b, i

            x_min, x_max = data[:, i].min() - 1, data[:, i].max() + 1
            y_min, y_max = data[:, i + 1].min() - 1, data[:, i + 1].max() + 1

            xx = np.arange(x_min, x_max, 0.1)
            yy = np.arange(y_min, y_max, 0.1)
            xv, yv = np.meshgrid(xx, yy)
            zv = predict(np.c_[xv.ravel(), yv.ravel()])
            zv = zv.reshape(xv.shape)

            self.ax[a][b].pcolormesh(xv, yv, zv, cmap=self.cmap_light)
            self.ax[a][b].scatter(data[:, i], data[:, i + 1], c=label)

        if img_save_path:
            self.fig.savefig(img_save_path)

    def show_pic(self, data, label, predict, img_save_path=None, *args, **kwargs):
        for a, b, i in self.draw_pic(data, label, predict, img_save_path, *args, **kwargs):
            pass

    def init_ani(self):
        self.ani_fig, self.ani_ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)
        self.ani_fig.set_tight_layout(True)
        self.ims = []

    def img_collections(self, data, label, w, bias, *args, **kwargs):
        for a, b, i, im in self.draw_ani(data, label, w, bias, *args, **kwargs):
            pass

    def draw_ani(self, data, label, w, bias, *args, **kwargs):
        im = []
        for i in range(self.n_features // 2):
            a = i // self.col
            b = i % self.col
            yield a, b, i, im

            x_min, x_max = data[:, i].min() - 1, data[:, i].max() + 1
            y_min, y_max = data[:, i + 1].min() - 1, data[:, i + 1].max() + 1

            self.ani_ax[a][b].set_xlim(x_min, x_max)
            self.ani_ax[a][b].set_ylim(y_min, y_max)

            xx = np.arange(x_min, x_max)
            y = -(w[0] * xx + bias) / w[1]

            line, = self.ani_ax[a][b].plot(xx, y, c='black', animated=True)
            im.append(line)
            im.append(self.ani_ax[a][b].scatter(data[:, i], data[:, i + 1], c=label, animated=True))

        self.ims.append(im)

    def show_ani(self, img_save_path=None, fps=10):
        ani = animation.ArtistAnimation(self.ani_fig, self.ims, interval=1000 // len(self.ims), blit=True,
                                        repeat_delay=1000, repeat=False)

        if img_save_path is not None:
            if img_save_path.endswith('.gif'):
                from matplotlib.animation import ImageMagickWriter as Writer
                ani.save(img_save_path, writer=Writer())
            else:
                from matplotlib.animation import FFMpegWriter as Writer
                ani.save(img_save_path, writer=Writer(fps=fps))

    def show(self):
        plt.show()


class Painter4cluster(Painter):
    def img_collections(self, data, label, *args, **kwargs):
        for a, b, i, im in self.draw_ani(data, label, *args, **kwargs):
            pass

    def draw_ani(self, data, label, *args, **kwargs):
        im = []
        for i in range(self.n_features // 2):
            a = i // self.col
            b = i % self.col
            im.append(self.ani_ax[a][b].scatter(data[:, i], data[:, i + 1], c=label, animated=True))

            yield a, b, i, im

        self.ims.append(im)

    def draw_pic(self, data, label, img_save_path=None, *args, **kwargs):
        for i in range(0, self.n_features, 2):
            a = i // self.col
            b = i % self.col
            self.ax[a][b].scatter(data[:, i], data[:, i + 1], c=label)
            yield a, b, i

        if img_save_path:
            self.fig.savefig(img_save_path)

    def show_pic(self, data, label, img_save_path=None, *args, **kwargs):
        for a, b, i in self.draw_pic(data, label, img_save_path, *args, **kwargs):
            pass


class Painter4decomposition:
    def beautify(self):
        import seaborn as sns
        sns.set(style="dark", palette="muted", color_codes=True)

    def init_pic(self):
        self.fig = plt.figure()
        self.fig.set_size_inches(8, 4)

    def show_pic(self, x, y, x_, img_save_path=None, *args, **kwargs):
        ax = self.fig.add_subplot(121, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap='rainbow')
        ax.view_init(4, -72)

        ax = self.fig.add_subplot(122)
        ax.scatter(x_[:, 0], x_[:, 1], c=y, cmap='rainbow')

        if img_save_path:
            self.fig.savefig(img_save_path)

    def show(self):
        plt.show()


def count_distances(x1, x2, method='euc', axis=None):
    # 欧氏距离
    if method == 'euc':
        return (np.sum((x1 - x2) ** 2, axis=axis)) ** 0.5
    # 余弦相似度
    elif method == 'cos':
        return np.dot(x1, x2) / np.sqrt((x1 ** 2).sum() * (x2 ** 2).sum())


def count_time(output='train complete!'):
    def wrap(train_func):
        def wrap2(*args, **kwargs):
            st = time.time()
            r = train_func(*args, **kwargs)
            et = time.time()
            t = et - st
            print(f'{output} time: {t}')
            return r

        return wrap2

    return wrap
