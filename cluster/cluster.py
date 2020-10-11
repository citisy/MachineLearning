import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)


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


class cluster:
    def __init__(self, data, k=3, show_img=False):
        self.data = np.array(data, dtype=float)
        self.k = k  # 簇数
        self.show_img = show_img
        self.n_features = self.data.shape[1]  # 数据维度
        self.n_samples = self.data.shape[0]  # 数据数量
        self.point_index = np.array(range(self.n_samples), dtype=int)  # 每一个点的归属簇

        if self.show_img:
            self.ims = []
            self.col = math.ceil(np.sqrt(self.n_features / 2))
            self.row = math.ceil(self.n_features / 2 / self.col)
            self.fig, self.ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)
            self.fig.set_tight_layout(True)

        self.norm()

    def norm(self):
        """
        normalize the data
        """
        pass

    @count_time
    def train(self, **kwargs):
        """
        可继承方法
        """
        return self.point_index

    # 求点与中心点的距离
    def get_r(self, x1, x2, method='euc'):
        # 欧氏距离
        if method == 'euc':
            return ((x1 - x2) ** 2).sum()
        # 余弦相似度
        elif method == 'cos':
            return -np.dot(x1, x2) / np.sqrt((x1 ** 2).sum() * (x2 ** 2).sum())

    def picture_collections(self, data, point_index, **kwargs):
        """
        Visualization of algorithms.
        """
        if not self.show_img:
            return

        im = []
        for i in range(self.n_features // 2):
            a = i // self.col
            b = i % self.col
            im.append(self.ax[a][b].scatter(data[:, i], data[:, i + 1], c=point_index, animated=True))

        self.ims.append(im)

    def show_ani(self, img_save_path):
        if self.show_img:
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=1000 // len(self.ims), blit=True,
                                            repeat_delay=1000, repeat=False)

            if img_save_path is not None:
                ani.save(img_save_path, writer='imagemagick')

            plt.show()

    def score(self):
        pass
