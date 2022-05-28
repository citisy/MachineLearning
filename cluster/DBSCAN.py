import numpy as np
from sklearn import datasets
from utils import Painter4cluster, count_distances


class DBSCAN:
    """
    　(1) 设定扫描半径 Eps, 并规定扫描半径内的密度值。若当前点的半径范围内密度大于等于设定密度值，则设置当前点为核心点；若某点刚好在某核心点的半径边缘上，则设定此点为边界点；若某点既不是核心点又不是边界点，则此点为噪声点。
    　(2) 删除噪声点。
    　(3) 将距离在扫描半径内的所有核心点赋予边进行连通。
    　(4) 每组连通的核心点标记为一个簇。
    　(5) 将所有边界点指定到与之对应的核心点的簇总。
    """

    def __init__(self, eps=None, min_samples=5, n_features=None, show_img=None, show_ani=None, painter=None):
        self.eps = eps
        self.min_samples = min_samples

        self.show_img = show_img
        self.show_ani = show_ani

        if self.show_img or self.show_ani:
            self.painter = painter or Painter4cluster(n_features)
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

            if not painter and self.show_ani:
                self.painter.init_ani()

    def fit_predict(self, data, img_save_path=None, ani_save_path=None):
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
        """
        n_samples = data.shape[0]
        n_features = data.shape[1]

        pred = np.arange(n_samples, dtype=int)

        point_index = [[] for _ in range(n_samples)]

        if self.eps is None:
            self.eps = self.detect_eps(data)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                r = count_distances(data[i], data[j])
                if r <= self.eps:
                    point_index[i].append(j)
                    point_index[j].append(i)

        # core point or border point
        points = set()
        for i in range(n_samples):
            if len(point_index[i]) >= self.min_samples:
                points.add(i)

                # core point范围内的点归为同一个簇
                # border point也会在这个时候被优化
                for j in point_index[i]:
                    if pred[i] == pred[j]:
                        continue

                    points.add(j)
                    ind_j = np.where(pred == pred[j])
                    pred[ind_j] = pred[i]

                    if self.show_ani:
                        self.painter.img_collections(data, pred)

        # noise point
        for i in range(n_samples):
            if i not in points:
                pred[i] = -1

        if self.show_ani:
            self.painter.img_collections(data, pred)

        # 重新编号
        idxes = []
        for k in np.unique(pred):
            idxes.append((k, pred == k))

        i = 0
        for k, idx in idxes:
            if k == -1:
                pred[idx] = -1
            else:
                pred[idx] = i
                i += 1

        if self.show_ani:
            self.painter.show_ani(ani_save_path, fps=25)

        if self.show_img:
            self.painter.show_pic(data, pred, img_save_path)
            self.painter.show()

        return pred

    def detect_eps(self, data):
        """根据输入数据猜测可能的eps值"""
        eps = 0
        for i in range(data.shape[0]):
            eps += np.sort(count_distances(data, data[i], axis=1))[self.min_samples * 2] / data.shape[0]

        return eps


def sample_test():
    x, y = datasets.make_moons(n_samples=500, noise=0.08)

    eps = 0
    min_samples = 5
    for i in range(x.shape[0]):
        eps += np.sort(count_distances(x, x[i], axis=1))[min_samples * 2] / x.shape[0]

    model = DBSCAN(eps=eps, min_samples=min_samples,
                   n_features=x.shape[1], show_ani=True, show_img=True
                   )
    pred = model.fit_predict(x,
                             # img_save_path='../img/DBSCAN.png',
                             # ani_save_path='../img/DBSCAN.mp4'
                             )

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.933025120973908"""


def real_data_test():
    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    eps = 0
    min_samples = 5
    for i in range(x.shape[0]):
        eps += np.sort(count_distances(x, x[i], axis=1))[2 * min_samples] / x.shape[0]

    model = DBSCAN(eps=eps, min_samples=min_samples)
    pred = model.fit_predict(x)

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.23389646019330715"""


def sklearn_test():
    from sklearn.cluster import DBSCAN

    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    eps = 0
    min_samples = 5
    for i in range(x.shape[0]):
        eps += np.sort(count_distances(x, x[i], axis=1))[2 * min_samples] / x.shape[0]

    model = DBSCAN(eps=eps, min_samples=min_samples)
    pred = model.fit_predict(x)

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.24121286330731062"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
    # sklearn_test()
