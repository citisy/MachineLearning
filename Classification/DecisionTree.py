import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter
import collections


class DT:
    """
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
        info(play)=-sum(p*log22(p))  method=entropy
                  =-sum(p^2)        method=gini

    2.  count the info(i) of outlook, i=sunny, overcast, or rainy,
        here only show how to count info(sunny):
        p(no|sunny)=3/5, p(yes|sunny)=2/5
        info(sunny)=-sum(p*log22(p))
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
            gini(outlook) = sum(pi*info(i))

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

    def __init__(self, n_feature=None, show_img=False):
        self.show_img = show_img
        if self.show_img:
            self.painter = Painter(n_feature)
            self.painter.init_pic()

    def pruning(self, tree, a=0):
        """todo: 剪枝算法"""
        pass

    def fit(self, data, label, max_depth=10, eps=-np.inf, is_seq=True, img_save_path=None):
        def recursive(x, y, i, depth=0):
            n_features = x.shape[1]

            # 计算节点的标签
            counter = collections.Counter(y)
            cl = max(counter.items(), key=lambda x: x[1])[0]

            if depth == max_depth:  # 达到最大迭代次数
                self.tree[i] = (None, None, max(counter.items(), key=lambda x: x[1])[0])
                return

            if len(counter) == 1:  # 所有实例属于同一类
                self.tree[i] = (None, None, cl)
                return

            # 计算信息增益最大的特征
            g = []
            for a in range(n_features):
                hd, hda, split_feature = self.get_gain(x[:, a], y, is_seq)
                g.append((a, hda, split_feature))

            select_feature_index, gain, split_feature = max(g, key=lambda x: x[1])

            # 计算划分点
            a = x[:, select_feature_index]

            self.tree[i] = (select_feature_index, split_feature, cl)

            if gain < eps:
                return

            # 递归建树
            if is_seq:
                xx, yy = x[a <= split_feature], y[a <= split_feature]
            else:
                xx, yy = x[a == split_feature], y[a == split_feature]

            if len(xx) and i * 2 < len(self.tree):
                recursive(xx, yy, 2 * i, depth + 1)

            if is_seq:
                xx, yy = x[a > split_feature], y[a > split_feature]
            else:
                xx, yy = x[a != split_feature], y[a != split_feature]

            if len(xx) and i * 2 + 1 < len(self.tree):
                recursive(xx, yy, 2 * i + 1, depth + 1)

        data = np.array(data, dtype=float)
        label = np.array(label)

        self.tree = [None] * (2 ** max_depth + 1)
        recursive(data, label, 1)

        if self.show_img:
            self.painter.show_pic(data, label, self.predict, img_save_path)
            self.painter.show()

    def get_gain(self, a, y, is_seq=True):
        """计算信息增益"""
        n_samples = len(a)

        counter = collections.Counter(y)
        hd = 0
        for k, v in counter.items():  # 计算先验概率
            py = v / n_samples
            hd -= self.get_info(py)

        counter2 = collections.Counter(a)
        g = []
        for k1, v1 in counter2.items():
            aa = a.copy()
            if is_seq:
                arg1, arg2 = aa <= k1, aa > k1
            else:
                arg1, arg2 = aa == k1, aa != k1

            aa[arg1], aa[arg2] = 1, 0  # 将特征变为01分布

            hda = self.get_da(aa, y, n_samples)

            g.append((k1, hda))

        split_feature, hda = min(g, key=lambda x: x[1])

        return hd, -hda, split_feature

    def get_info(self, p):
        return p * np.log2(p)

    def get_da(self, a, y, n_samples, init_di=0):
        counter2 = collections.Counter(a)
        da = 0
        for k1, v1 in counter2.items():
            px = v1 / n_samples

            yi = y[a == k1]
            n = len(yi)

            counter3 = collections.Counter(yi)
            di = init_di
            for k3, v3 in counter3.items():
                p = v3 / n
                di -= self.get_info(p)

            da += px * di

        return da

    def predict(self, x):
        x = np.array(x, dtype=float)
        n_test = x.shape[0]

        pre = np.zeros(n_test, dtype=int)

        for a in tqdm(range(n_test)):
            i = 1
            xx = x[a]
            while True:
                if i > len(self.tree) or not self.tree[i]:
                    i //= 2
                    pre[a] = self.tree[i][2]
                    break

                select_feature_index, split_feature, cl = self.tree[i]

                if select_feature_index is None:
                    pre[a] = cl
                    break

                if xx[select_feature_index] <= split_feature:
                    i *= 2

                else:
                    i = i * 2 + 1

        return pre


ID3 = DT


class C45(DT):
    def fit(self, data, label, max_depth=10, eps=0.01, is_seq=True, img_save_path=None):
        super(C45, self).fit(data, label, max_depth, eps, is_seq, img_save_path)

    def get_gain(self, a, y, is_seq=True):
        hd, hda, split_feature = super(C45, self).get_gain(a, y, is_seq)
        hda = (hd + hda) / hd
        return hd, hda, split_feature


class Cart(DT):
    def get_da(self, a, y, n_samples, init_di=1):
        return super(Cart, self).get_da(a, y, n_samples, init_di)

    def get_info(self, p):
        return p ** 2


def make_weather_data():
    data = [['sunny', 'hot', 'high', 'FALSE', 'no'],
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
    return np.array(data), labels


def id3_test():
    x, y = datasets.make_blobs(centers=5, n_samples=500)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = ID3(x.shape[1], show_img=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/DT_id3.png',
              )

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.98"""
    # todo: 过拟合


def real_data_id3_test():
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = ID3()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9473684210526315"""


def c45_test():
    x, y = datasets.make_blobs(centers=5, n_samples=500)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = C45(x.shape[1], show_img=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/DT_c45.png',
              )

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.98"""


def real_data_c45_test():
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = C45()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9473684210526315"""


def cart_test():
    x, y = datasets.make_blobs(centers=5, n_samples=500)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Cart(x.shape[1], show_img=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/DT_cart.png',
              )

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.98"""


def real_data_cart_test():
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Cart()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9035087719298246"""


def sklearn_test():
    from sklearn.tree import DecisionTreeClassifier

    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.8859649122807017"""


if __name__ == '__main__':
    # id3_test()
    # real_data_id3_test()

    # c45_test()
    # real_data_c45_test()

    # cart_test()
    real_data_cart_test()

    # sklearn_test()
