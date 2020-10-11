"""
　(1) 设定扫描半径 Eps, 并规定扫描半径内的密度值。若当前点的半径范围内密度大于等于设定密度值，则设置当前点为核心点；若某点刚好在某核心点的半径边缘上，则设定此点为边界点；若某点既不是核心点又不是边界点，则此点为噪声点。
　(2) 删除噪声点。
　(3) 将距离在扫描半径内的所有核心点赋予边进行连通。
　(4) 每组连通的核心点标记为一个簇。
　(5) 将所有边界点指定到与之对应的核心点的簇总。
"""

from cluster import *


class DBSCAN(cluster):
    def __init__(self, data, show_img=False, eps=0.01, threshold=3):
        self.eps = eps
        self.threshold = threshold
        super(DBSCAN, self).__init__(data, show_img=show_img)

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
        """
        eg:
            index: list type
                    [0,1,2] [3,4,5]
            distances: list type
                       10      0.4
                      0.4      10
             there is 2 class according to the list of distances,
             0,1,2 is in a class and 3,4,5 is in another class,
             we know that, the list of distances is symmetry,
             u can only use half of list distances
             for changing the code by yourselves, it's very easy(smile).
        """
        distances = [[10 for _ in range(self.n_samples)] for __ in range(self.n_samples)]
        group_list = [{i} for i in range(self.n_samples)]

        self.cent_cn = np.zeros(self.n_samples, dtype=int)
        point_index = [[] for _ in range(self.n_samples)]
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                r = self.get_r(self.data[i], self.data[j])
                distances[i][j] = r
                if r <= self.eps:
                    self.cent_cn[i] += 1
                    self.cent_cn[j] += 1
                    point_index[i].append(j)

        # core point
        core_point = []
        for i in range(self.n_samples):
            if self.cent_cn[i] >= self.threshold:
                core_point.append(i)

                for j in point_index[i]:
                    x = y = -1
                    for k in range(len(group_list)):  # find data[i] and data[j] in eps
                        if i in group_list[k]:
                            x = k
                        if j in group_list[k]:
                            y = k

                    if all([x == y, x != -1, y != -1]):  # x and y in the same class
                        continue

                    group_list[x] |= group_list[y]
                    for k in group_list[x]:
                        self.point_index[k] = i  # we label the class with data[i]'s index

                    # update distances mapping
                    for k in range(len(distances)):
                        if x == k:
                            continue
                        distances[x][k] = min(distances[x][k], distances[y][k])

                    for k in range(len(distances)):
                        del distances[k][y]

                    del distances[y]
                    del group_list[y]

                    self.picture_collections(self.data, self.point_index)

        # border point
        for i in range(self.n_samples):
            if 0 < self.cent_cn[i] < self.threshold:
                x = -1
                for k in range(len(group_list)):
                    if i in group_list[k]:
                        x = k

                if x != -1 and len(group_list[x]) > 1:  # it is core point class
                    continue

                argsort = np.argsort(distances[x])  # sort by distances to other class
                j = 0
                while j < len(group_list):
                    y = argsort[j]
                    if len(group_list[y]) > 1:  # find the core point class
                        group_list[x] |= group_list[y]
                        for k in group_list[x]:
                            self.point_index[k] = i

                        for k in range(len(distances)):
                            distances[x][k] = min(distances[x][k], distances[y][k])

                        for k in range(len(distances)):
                            del distances[k][y]

                        del distances[y]
                        del group_list[y]
                        break
                    j += 1

        self.picture_collections(self.data, self.point_index)

        # noise point
        for i in range(self.n_samples):
            if self.cent_cn[i] == 0:
                self.point_index[i] = -1

        self.picture_collections(self.data, self.point_index)

        self.show_ani(kwargs.get('img_save_path'))

        return self.point_index


def sklearn_DBSCAN(x, eps, threshold):
    from sklearn.cluster import DBSCAN

    y_pred = DBSCAN(eps=eps, min_samples=threshold).fit(x)

    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()


if __name__ == '__main__':
    from sklearn.datasets import make_moons as make_data

    np.random.seed(6)
    x, y = make_data(n_samples=500, noise=0.08)

    eps, threshold = 0.01, 3
    model = DBSCAN(x, show_img=True, eps=eps, threshold=threshold)
    model.train()
    # model.train(img_save_path='../img/kmeans.gif')
