"""
　　(1) 任意选择k个对象作为初始的簇中心；
　　(2) repeat；
　　(3) 根据簇中对象的平均值，将每个对象(重新)赋予最类似的簇；
　　(4) 更新簇的平均值，即计算每个簇中对象的平均值；
　　(5) until不再发生变化。
"""
from cluster import *


class Kmeans(cluster):
    def norm(self):
        self.cent = np.zeros((self.k, self.n_features))     # 中心点集

        for i in range(self.n_features):
            amax = self.data[:, i].max()
            amin = self.data[:, i].min()

            # 数据归一化
            self.data[:, i] /= amax

            # 随机生成中心点落在数据中
            self.cent[:, i] = np.random.random(self.k) * (amax - amin) / amax + amin / amax

    # 模型训练
    @count_time
    def train(self, **kwargs):
        for _ in range(kwargs.get('itera') or 100):
            self.cent_cn = np.zeros(self.k, dtype=int)  # 每个簇包含点的数量
            for a in range(self.n_samples):
                r = []

                # 求每个点与每个中心点的距离
                for b in range(self.k):
                    r.append(self.get_r(self.data[a], self.cent[b]))

                # 离点最近的中心点的下标
                max_ind = np.argmin(r)

                # 标记归属簇
                self.point_index[a] = max_ind

                self.cent_cn[max_ind] += 1

            new_cent = np.zeros((self.k, self.n_features))

            # 遍历所有数据，更新中心点位置
            for a in range(self.n_samples):
                new_cent[self.point_index[a]] += self.data[a]

            # 如果某一个簇包含0个点，说明这个簇的中心点选择有问题，需要重新选择中心点
            for a in range(self.k):
                if self.cent_cn[a] == 0:
                    new_cent[a] = self.cent[a] + np.random.random(self.n_features) - 0.5
                else:
                    new_cent[a] /= self.cent_cn[a]

            # 中心点位置不变，训练完成，退出循环
            if (self.cent == new_cent).all():
                break

            self.cent = new_cent

            self.img_collections(self.data, self.point_index, cent=self.cent)

        self.show_gif(kwargs.get('img_save_path'))

        return self.point_index

    def img_collections(self, data, point_index, **kwargs):
        if not self.show_img:
            return

        cent = kwargs.get('cent')
        im = []
        for i in range(self.n_features // 2):
            a = i // self.col
            b = i % self.col

            # 原数据
            im.append(self.ax[a][b].scatter(data[:, i], data[:, i + 1], c=point_index, animated=True))

            # 中心点
            im.append(self.ax[a][b].scatter(cent[:, i], cent[:, i + 1], c='r', animated=True))

        self.ims.append(im)


if __name__ == '__main__':
    from sklearn import datasets

    X, y = datasets.make_blobs(n_samples=500, n_features=2, random_state=3)
    model = Kmeans(X, 3, show_img=True)
    model.train()
    # model.train(img_save_path='../img/kmeans.gif')

    # print(model.score())
    # y_pred = MiniBatchKMeans(n_clusters=2, batch_size=200, random_state=9).fit_predict(X)
