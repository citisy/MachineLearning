"""
　(1) 将每个对象看作一类，计算两两之间的最小距离；
　(2) 将距离最小的两个类合并成一个新类；
　(3) 重新计算新类与所有类之间的距离；
　(4) 重复(2)、(3)，直到所有类最后合并成一类
"""

from cluster import *


class Hierarchical(cluster):
    def norm(self):
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
            amax = abs(self.data[:, i].max())
            amin = abs(self.data[:, i].min())
            self.data[:, i] /= max(amax, amin) * 1.2

    @count_time
    def train(self, **kwargs):
        distances = np.zeros((self.n_samples, self.n_samples))
        # 两点距离最大为根号2，设为10可视为无限远，即为忽略的点
        distances += 10

        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                distances[i][j] = self.get_r(self.data[i], self.data[j])

        for _ in range(self.n_samples - self.k):
            min_ind = np.argmin(distances)
            j = min_ind % self.n_samples
            i = min_ind // self.n_samples

            # 找簇
            ind_j = np.where(self.point_index == self.point_index[j])
            ind_i = np.where(self.point_index == self.point_index[i])

            # 距离最近的簇只保留一个
            self.point_index[ind_j] = self.point_index[i]

            # 簇内成员距离都为10
            for i in ind_i[0]:
                for j in ind_j[0]:
                    distances[min(i, j)][max(i, j)] = 10

            if _ % 5 == 4:
                self.picture_collections(self.data, self.point_index)

        self.show_ani(kwargs.get('img_save_path'))

        return self.point_index


def sklearn_hierarchical(x):
    from sklearn.cluster import AgglomerativeClustering

    y_pred = AgglomerativeClustering(n_clusters=2).fit(x)

    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()


if __name__ == '__main__':
    from sklearn.datasets import make_moons as make_data

    np.random.seed(6)
    x, y = make_data(n_samples=502, noise=0.08)

    model = Hierarchical(x, 2, show_img=True)
    model.train()
    # model.train(img_save_path='../img/Hierarchical.gif')
