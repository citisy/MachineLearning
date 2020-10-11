import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

id3 = 1
c45 = 2
cart = 3
entropy = 1
gini = 2


class DT:
    """
    https://blog.csdn.net/qq_36330643/article/details/77415451
    https://www.cnblogs.com/en-heng/p/5013995.html

    there is a dataset:
        outlook	    temperature	humidity	windy	play
        sunny	    hot	        high	    FALSE	no
        sunny	    hot	        high	    TRUE	no
        overcast	hot	        high	    FALSE	yes
        rainy	    mild	    high	    FALSE	yes
        rainy	    cool	    normal	    FALSE	yes
        rainy	    cool	    normal	    TRUE	no
        overcast	cool	    normal	    TRUE	yes
        sunny	    mild	    high	    FALSE	no
        sunny	    cool	    normal	    FALSE	yes
        rainy	    mild	    normal	    FALSE	yes
        sunny	    mild	    normal	    TRUE	yes
        overcast	mild	    high	    TRUE	yes
        overcast	hot	        normal	    FALSE	yes
        rainy	    mild	    high	    TRUE	no

    'outlook' is feature
    'sunny' is status of features
    'play' is label
    we set 'no' is neg(-) samples, 'yes' is pos(+) samples

    1.  count the info(play), Entropy of y:
        p(no)=5/14, p(yes)=9/14
        info(play)=-sum(p*log2(p))  method=entropy
                  =-sum(p^2)        method=gini

    2.  count the info(i) of outlook, i=sunny, overcast, or rainy,
        here only show how to count info(sunny):
        p(no|sunny)=3/5, p(yes|sunny)=2/5
        info(sunny)=-sum(p*log2(p))
        status of other features count like that

    3.  count the gain(outlook), gain(temperature), gain(humidity), gain(windy)
        here only show how to count gain(outlook):
        p(sunny)=5/14, p(overcast)=4/14, p(rainy)=5/14
        if it is id3:
            gain(outlook) = info(play)-sum(pi*info(i))
        if it is c45:
            split_info(play) = info(p(status|outputs))
            gain(outlook) = (info(play)-sum(pi*info(i)))/split_info(play)
        if it is cart:
            gain(outlook) = sum(pi*info(i))

    4.  count max(gain(i))(min(gain(i)) if it is cart),
        we get the feature of max gain, let it be the root,
        and let the status of the features divide to the children
        here, if the the status is too many, we can set some threshold,
        divide flags is info(status) of the feature

    5.  let the sample(play)=sample(status), info(play)=info(status),
        repeat the 2~4, recurse to build the tree until:
        * it's max depth
        * the feature is empty
        * all the samples have the same output
    """

    def __init__(self, data, label, feature_idx=None, method=id3, max_depth=0,
                 is_seq=True, show_img=False, img_save_path=None):
        self.img_save_path = img_save_path

        data_ = np.array(data)
        label = np.array(label)
        n_samples = data_.shape[0]
        n_features = data_.shape[1]
        if feature_idx is not None:
            self.feature_idx = np.array(feature_idx)
        else:
            self.feature_idx = np.array(range(n_features))
        self.is_seq = is_seq  # 是否为序列

        if is_seq:
            data_ = self.norm(data_)

        outputs = np.unique(label)
        p = [np.sum(label == i) / n_samples for i in outputs]
        info_d = self.get_info(p)

        tree = {}
        self.train(data_, label, info_d, tree, max_depth,
                   feature_idx=self.feature_idx, is_seq=is_seq, method=method)
        self.tree = tree

        if show_img:
            self.show(data, label)

    def norm(self, x):
        """
        eg:
            1.1 -> 1.0
            2.8 -> 2.0
        """
        return np.array(np.int_(x), dtype=float)

    def train(self, data, label, info_d, tree, max_depth=0,
              depth=0, feature_idx=None, is_seq=True, method=id3):
        outputs = np.unique(label)
        n_features = data.shape[1]
        n_samples = data.shape[0]

        if len(outputs) == 1:  # 所有实例属于同一类
            tree['output'] = outputs[0]
            return
        elif data.size == 0:  # 特征值为空集，有两种情况
            if n_samples == 0:
                tree['output'] = None
            else:
                tree['output'] = collections.Counter(label).most_common(1)[0][0]
            return
        elif depth >= max_depth:  # 达到最大迭代次数
            tree['output'] = collections.Counter(label).most_common(1)[0][0]
            return

        # 计算信息熵
        info_ij = []
        statuses = []
        p_i = [np.sum(label == i) / n_samples for i in outputs]  # 数据集中每个类的概率

        for i in range(n_features):
            info_ij.append([])
            status = np.unique(data[:, i])
            statuses.append(status)
            for j in status:
                idx = np.where(data[:, i] == j)
                n = len(idx[0])
                p = [np.sum(label[idx] == k) / n for k in outputs]  # 特征集的每个状态中每个类的概率
                if method == id3:
                    info_ij[-1].append((n / n_samples, self.get_info(p), j))  # (pj, info_j, j)
                elif method == c45:
                    p_split = [p[_] / p_i[_] for _ in range(len(p))]
                    info_ij[-1].append(
                        (n / n_samples, self.get_info(p), j, self.get_info(p_split)))  # (pj, info_j, j, info_split)
                elif method == cart:
                    info_ij[-1].append((n / n_samples, self.get_info(p, method=gini), j))  # (pj, info_j, j)

        # count gain(features)
        gain_i = []
        for i in range(len(info_ij)):
            _ = info_d
            for j in info_ij[i]:
                if method == id3:
                    _ -= j[0] * j[1]
                elif method == c45:
                    _ = (_ - j[0] * j[1] + 1) / (j[3] + 1)
                elif method == cart:
                    _ += j[0] * j[1]
            gain_i.append(_)

        argsort = np.argsort(gain_i)

        # gini取最小值，entropy取最大值
        if method == cart:
            flag = 0
        else:
            flag = -1

        hit_feature_index = argsort[flag]
        hit_feature = feature_idx[hit_feature_index]
        hit_feature_info = info_ij[hit_feature_index]
        tree[hit_feature] = {}
        if is_seq:
            hit_status_index = int(np.argmax([_[1] for _ in hit_feature_info]))
            max_info = hit_feature_info[hit_status_index][1]
            _ = []
            info = 0
            for i in hit_feature_info:
                if i[1] == max_info:
                    _.append(i[2])
                    info += i[1]

            divide = np.median(_)   # todo: 平均值或其他统计方法
            tree[hit_feature][(-np.inf, divide)] = {}   # 二叉树
            left_part = np.where(data[:, hit_feature_index] <= divide)

            self.train(data[left_part], label[left_part], info, tree[hit_feature][(-np.inf, divide)],
                       max_depth, depth + 1, feature_idx, is_seq, method=method)

            tree[hit_feature][(divide, np.inf)] = {}
            right_part = np.where(data[:, hit_feature_index] > divide)
            self.train(data[right_part], label[right_part], info, tree[hit_feature][(divide, np.inf)],
                       max_depth, depth + 1, feature_idx, is_seq, method=method)

        else:
            argsort = argsort[np.where(argsort != argsort[flag])]
            for i, status in enumerate(statuses[hit_feature_index]):
                tree[hit_feature][(status,)] = {}
                hit_status_index = np.where(data[:, hit_feature_index] == status)

                self.train(data[hit_status_index][:, argsort], label[hit_status_index],
                           hit_feature_info[i][1], tree[hit_feature][(status,)],
                           max_depth, depth + 1, feature_idx[argsort], is_seq, method=method)

    def pruning(self, tree, a=0):
        pass

    def get_info(self, p, method=entropy):
        """
        return info, gain of p
            info = -SUM(p*log2(p)), if p = 0 , p*log2(p) = 0, method = entropy
                 = -SUM(p^2),   method = gini
        """
        info = 0
        if method == entropy:
            for i in p:
                if i == 0:
                    continue
                info -= i * np.log2(i)
        elif method == gini:
            info = 1
            for i in p:
                info -= i ** 2
        return info

    def predict(self, data):
        data = np.array(data)
        pre = []
        for pre_data in data:
            d = self.tree
            while 'output' not in d:
                for k1, v1 in d.items():  # k1: hit_feature
                    for k2, v2 in v1.items():  # k2: classification rules v2: label
                        if self.is_seq:
                            if k2[0] < pre_data[self.feature_idx == k1] <= k2[1]:
                                d = v2
                        else:
                            if pre_data[self.feature_idx == k1] in k2:
                                d = v2
            pre.append(d['output'])
        return np.array(pre)

    def show(self, x, y):
        fig, ax = plt.subplots()

        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

        xx = np.arange(x_min, x_max, 0.1)
        yy = np.arange(y_min, y_max, 0.1)
        xv, yv = np.meshgrid(xx, yy)
        zv = self.predict(np.c_[xv.ravel(), yv.ravel()])
        zv = zv.reshape(xv.shape)

        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AAAAAA', '#FFFFFF'])
        ax.pcolormesh(xv, yv, zv, cmap=cmap_light)
        ax.scatter(x[:, 0], x[:, 1], c=y)

        if self.img_save_path:
            fig.savefig(self.img_save_path)

        plt.show()


def make_weather_data():
    datasets = [['sunny', 'hot', 'high', 'FALSE', 'no'],
                ['sunny', 'hot', 'high', 'TRUE', 'no'],
                ['overcast', 'hot', 'high', 'FALSE', 'yes'],
                ['rainy', 'mild', 'high', 'FALSE', 'yes'],
                ['rainy', 'cool', 'normal', 'FALSE', 'yes'],
                ['rainy', 'cool', 'normal', 'TRUE', 'no'],
                ['overcast', 'cool', 'normal', 'TRUE', 'yes'],
                ['sunny', 'mild', 'high', 'FALSE', 'no'],
                ['sunny', 'cool', 'normal', 'FALSE', 'yes'],
                ['rainy', 'mild', 'normal', 'FALSE', 'yes'],
                ['sunny', 'mild', 'normal', 'TRUE', 'yes'],
                ['overcast', 'mild', 'high', 'TRUE', 'yes'],
                ['overcast', 'hot', 'normal', 'FALSE', 'yes'],
                ['rainy', 'mild', 'high', 'TRUE', 'no']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return np.array(datasets), labels


def sklearn_DT(x, y):
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    model.fit(x, y)

    fig, ax = plt.subplots()

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx = np.arange(x_min, x_max, 0.1)
    yy = np.arange(y_min, y_max, 0.1)
    xv, yv = np.meshgrid(xx, yy)

    zv = model.predict(np.c_[xv.ravel(), yv.ravel()])
    zv = zv.reshape(xv.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AAAAAA', '#FFFFFF'])
    ax.pcolormesh(xv, yv, zv, cmap=cmap_light)
    ax.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


if __name__ == '__main__':
    from sklearn.datasets import make_blobs as make_data

    np.random.seed(6)

    # data, feature_idx = make_weather_data()
    # x, y = data[:, :-1], data[:, -1:]
    # y = np.array(y).reshape((-1,))
    # model = DT(x, y, feature_idx, is_seq=False, method=id3, max_depth=10)
    # print(model.tree)
    # print(y == model.predict(x))

    x, y = make_data(n_samples=500, n_features=2, centers=5)
    # img_save_path = '../img/id3.png'
    img_save_path = None
    model = DT(x, y, is_seq=True, method=cart, max_depth=10,
               show_img=True, img_save_path=img_save_path)
    print(model.tree)
