# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import collections

"""
id3
c4.5
cart
"""


class DT(object):
    def __init__(self, data, label):
        self.data = np.array(data)      # data -> [n_samples, n_features]
        self.label = label
        self.n_samples = len(self.label)
        dict_ = collections.Counter(self.label)
        self.dict_ = {}
        for (k, v) in dict_.items():
            self.dict_[k] = {}
            self.dict_[k]['index'] = np.where(self.label == k)
            self.dict_[k]['count'] = v
            self.dict_[k]['py'] = v / self.n_samples
        self.n_features = len(self.dict_)

    def id3(self, x):
        n_features = self.data.shape[1]
        p = []
        for v in self.dict_.values():
            p.append(v['py'])
        info_d = self.get_info(p)
        self.gain_i = np.zeros(n_features)
        for a in range(n_features):
            features = np.unique(self.data[:, a])
            info_i = 0
            p = []
            for i in features:
                for v in self.dict_.values():
                    n = np.sum(self.data[v['index'], a] == i)
                    p.append(n / v['count'])
                info_dj = self.get_info(p)
                dj_div_d = np.sum(self.data[:, a] == i) / self.n_samples
                info_i += dj_div_d * info_dj
            self.gain_i[a] = info_d - info_i
        self.sort = self.gain_i.argsort()
        return self.make_trees(x)

    def c45(self):
        n_features = self.data.shape[1]
        p = []
        for v in self.dict_.values():
            p.append(v['py'])
        info_d = self.get_info(p)
        self.gain_radio = np.zeros(n_features)
        for a in range(n_features):
            features = np.unique(self.data[:, a])
            info_i = 0
            p = []
            for i in features:
                n = np.sum(self.data[:, a] == i)
                p.append(n / self.n_samples)
            info_split = self.get_info(p)
            self.gain_radio[a] = (info_d - info_i) / info_split
        self.sort = self.gain_radio.argsort()
        return self.make_trees(x)

    # 一种分类回归树
    def cart(self):
        pass

    def make_trees(self, x):
        n_sample = len(x)
        pre = np.zeros(n_sample, dtype=int)
        for a in range(n_sample):
            data = np.concatenate((self.data, self.label.reshape(-1, 1)), axis=1)   # data -> [n_samples, n_features+1]
            # 从最大的信息增益处开始检索，遇到没有的元素则跳过继续往下检索
            for b in self.sort:
                if x[a][b] in data[:, b]:
                    index = np.where(data[:, b] == x[a][b])
                    data = data[index]
                else:
                    continue
            pre[a] = data[np.random.randint(data.shape[0])][-1]
        return pre

    def get_info(self, p):
        """
        return info, gain of p
            info = -SUM(p*log2(p)), if p = 0 , p*log2(p) = 0
        """
        info = 0
        for i in p:
            if i == 0:
                continue
            info -= i * np.log2(i)
        return info


if __name__ == '__main__':
    x = np.random.randint(5, size=(100, 10))
    y = np.random.randint(5, size=(100,))

    model = DT(x, y)
    pre = model.id3(x)
    # 预测与样本标签可能不一致，因为我们训练的数据是随机的，可能有多于一条相同的x指向不同的y
    # x特征数越大，出现上述情况的概率越小
    for i in range(100):
        print(pre[i], y[i])
