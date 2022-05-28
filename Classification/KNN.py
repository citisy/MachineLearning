import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter
import collections
from tqdm import tqdm


class KNN:
    def __init__(self, n_features=None, show_img=False):
        self.show_img = show_img

        if self.show_img:
            self.painter = Painter(n_features)
            self.painter.beautify()
            self.painter.init_pic()

    def fit(self, data, label, k=10, **kwargs):
        """knn是lazy learning，基本不学习，网络结构很简单，所以该方法只是简单保存传进来参数，并没有任何运算
        判定标准：“近朱者赤，近墨者黑”以及“少数服从多数”。
        缺点：
            1、每次预测时都有遍历所有样本计算距离，所以预测的计算量很大
            2、小数据的预测错误率高
        """
        self.data = np.array(data, dtype=float)
        self.label = np.array(label)
        self.k = k
        self.n_sample = self.data.shape[0]

        if self.show_img:
            self.painter.show_pic(self.data, self.label, self.predict, **kwargs)
            self.painter.show()

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]
        pre = np.zeros(n_test, dtype=int)

        for a in tqdm(range(n_test)):
            r = np.zeros(self.n_sample)

            # 计算预测点到每个样本的距离
            for i in range(self.n_sample):
                r[i] = self.cal_distance(x[a], self.data[i])

            # 统计预测点附近k个点的标签，取出现次数最多的标签
            argsort = r.argsort()
            pre[a] = collections.Counter(self.label[argsort[:self.k]]).most_common(1)[0][0]

        return pre

    def cal_distance(self, a, b):
        return np.linalg.norm(a - b)


class KdTree(KNN):
    """使用了kd树的KNN算法"""

    def fit(self, data, label, k=10, **kwargs):
        self.data = np.array(data, dtype=float)
        self.label = np.array(label)
        self.k = k
        self.n_sample = self.data.shape[0]
        n_feature = self.data.shape[1]

        self.tree = [None for _ in range(2 ** int(np.ceil(np.log2(self.n_sample))) + 1)]

        def recursive(data, label, dim, i):
            dim = dim if dim < n_feature else 0

            arg = data[:, dim].argsort()
            mid = arg[data.shape[0] // 2]  # 中位数坐标
            self.tree[i] = (dim, label[mid], data[mid])  # (dim, label, split_point)
            if data.shape[0] > 1:
                recursive(data[arg[:data.shape[0] // 2]], label[arg[:data.shape[0] // 2]],
                          dim + 1, i * 2)
            if data.shape[0] > 2:
                recursive(data[arg[data.shape[0] // 2 + 1:]], label[arg[data.shape[0] // 2 + 1:]],
                          dim + 1, i * 2 + 1)

        recursive(self.data, self.label, 0, 1)
        if self.show_img:
            self.painter.show_pic(self.data, self.label, self.predict, **kwargs)
            self.painter.show()

    def predict(self, x):
        assert len(self.tree) > 1, 'No training data!'

        x = np.array(x)

        n_test = x.shape[0]
        pre = np.zeros(n_test, dtype=int)

        for a in tqdm(range(n_test)):
            i = 1
            k = self.k

            while i < len(self.tree) and self.tree[i]:  # 找叶子节点
                dim, _, point = self.tree[i]
                if x[a][dim] < point[dim]:
                    i *= 2
                else:
                    i = i * 2 + 1

            i //= 2

            cache_a, cache_b, cache_c = [], [], set()

            while i > 0:
                dim, label, point = self.tree[i]

                if i not in cache_c:  # 当前节点未被遍历过
                    dis = self.cal_distance(x[a], point)

                    cache_c.add(i)

                    if k:
                        cache_a.append(dis)
                        cache_b.append(label)
                        k -= 1

                    else:
                        min_dis = max(cache_a)
                        if dis < min_dis:
                            idx = cache_a.index(min_dis)
                            cache_a[idx] = dis
                            cache_b[idx] = label

                min_dis = max(cache_a)

                if (
                        2 * i not in cache_c
                        and 2 * i < len(self.tree)
                        and self.tree[2 * i]
                        and x[a][dim] - point[dim] < min_dis
                ):  # 遍历左节点
                    i = 2 * i

                elif (
                        2 * i + 1 not in cache_c
                        and 2 * i + 1 < len(self.tree)
                        and self.tree[2 * i + 1]
                        and point[dim] - x[a][dim] < min_dis
                ):  # 遍历右节点
                    i = 2 * i + 1

                else:  # 回退父节点
                    i //= 2

            pre[a] = collections.Counter(cache_b).most_common(1)[0][0]

        return pre


def sample_test():
    x, y = datasets.make_blobs(centers=5, n_samples=200)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = KdTree(x.shape[1], show_img=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/KNN.png'
              )

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 1.0"""


def real_data_test():
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = KdTree()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.956140350877193"""


def sklearn_test():
    from sklearn.neighbors import KNeighborsClassifier

    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9385964912280702"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
    # sklearn_test()
