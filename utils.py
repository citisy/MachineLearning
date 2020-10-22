import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from sklearn import datasets
from sklearn.model_selection import train_test_split

np.random.seed(6)


class Painter:
    def __init__(self, n_features):
        self.n_features = n_features

        self.cmap_light = ListedColormap(['#AAFFAA', '#AAAAFF', '#FFFFAA', '#AAAAAA', '#FFFFFF', '#FFAAAA'])

        self.col = int(np.ceil(np.sqrt(n_features / 2)))
        self.row = int(np.ceil(n_features / 2 / self.col))
        self.fig, self.ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)

    def beautify(self):
        import seaborn as sns
        sns.set(style="white", palette="muted", color_codes=True)

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

    def img_collections(self, data, label, *args, **kwargs):
        im = []
        for i in range(self.n_features // 2):
            a = i // self.col
            b = i % self.col
            yield a, b, i, im

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
