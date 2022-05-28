import numpy as np
from tqdm import tqdm
from sklearn import datasets, metrics
from utils import Painter4cluster, count_distances


class Hierarchical:
    """
    　(1) 将每个对象看作一类，计算两两之间的最小距离；
    　(2) 将距离最小的两个类合并成一个新类；
    　(3) 重新计算新类与所有类之间的距离；
    　(4) 重复(2)、(3)，直到所有类最后合并成一类
    """

    def __init__(self, n_clusters, n_features=None, show_img=None, show_ani=None, painter=None):
        self.n_clusters = n_clusters

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
        """自顶向下的分裂方法"""
        n_samples = data.shape[0]
        n_features = data.shape[1]
        pred = np.arange(n_samples)
        distances = np.zeros((n_samples, n_samples)) + np.inf  # 初始各个点与点间的距离为无穷远

        # 因为矩阵是对称的，所以只计算一半矩阵即可
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i][j] = count_distances(data[i], data[j])

        for _ in tqdm(range(n_samples - self.n_clusters)):
            min_ind = np.argmin(distances)
            j = min_ind % n_samples
            i = min_ind // n_samples

            # 找簇
            ind_j = np.where(pred == pred[j])
            ind_i = np.where(pred == pred[i])

            # 距离最近的簇只保留一个
            # 这里一开始使用set合并的方法，但需要不断进行list的pop操作，消耗资源较大，后来改成改变标签纸值的方式
            pred[ind_j] = pred[i]

            # 簇内距离变为无穷远
            for i in ind_i[0]:
                for j in ind_j[0]:
                    distances[min(i, j)][max(i, j)] = np.inf

            if _ % 5 == 4 and self.show_ani:
                self.painter.img_collections(data, pred)

        # 重新编号
        idxes = []
        for k in np.unique(pred):
            idxes.append(pred == k)

        for i, idx in enumerate(idxes):
            pred[idx] = i

        if self.show_ani:
            self.painter.show_ani(ani_save_path, fps=5)

        if self.show_img:
            self.painter.show_pic(data, pred, img_save_path)
            self.painter.show()

        return pred


def sample_test():
    x, y = datasets.make_moons(n_samples=502, noise=0.08)

    model = Hierarchical(n_clusters=2,
                         n_features=x.shape[1], show_ani=True, show_img=True
                         )
    pred = model.fit_predict(x,
                             # img_save_path='../img/Hierarchical.png',
                             # ani_save_path='../img/Hierarchical.mp4'
                             )

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 1.0"""


def real_data_test():
    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    model = Hierarchical(n_clusters=3)
    pred = model.fit_predict(x)

    print(y)
    print(pred)

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.005443835443708631"""


def sklearn_test():
    from sklearn.cluster import AgglomerativeClustering

    dataset = datasets.load_wine()

    x, y = dataset.data, dataset.target

    model = AgglomerativeClustering(n_clusters=3)
    pred = model.fit_predict(x)

    print('ARI:', metrics.adjusted_rand_score(y, pred))
    """ARI: 0.36840191587483156"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
    # sklearn_test()
