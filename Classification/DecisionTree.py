"""
https://blog.csdn.net/qq_36330643/article/details/77415451
https://www.cnblogs.com/en-heng/p/5013995.html
"""

import numpy as np
import json
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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

    def __init__(self, data_, label, method='id3', max_depth=np.inf,
                 is_seq=1, draw=0, is_save=0, save_fn='save.png'):
        self.dic = {}
        data = np.array(data_)
        label = np.array(label).reshape((-1,))
        n_samples = data.shape[0]
        n_features = data.shape[1]
        self.is_seq = is_seq
        if is_seq:
            data = self.norm(data)

        outputs = np.unique(label)
        p = [np.sum(label == i) / n_samples for i in outputs]
        info_d = self.get_info(p)
        dic = {}
        self.recurse(data, label, info_d, dic, max_depth,
                     feature_idx=np.array(range(n_features)), is_seq=is_seq, method=method)
        self.dic = dic
        if draw:
            self.show(data_, label, is_save, save_fn=save_fn)

    def norm(self, x):
        """
        rules of normalize:
            eg:
                1.1 -> 1.0
                2.8 -> 2.0
        """
        return np.array(np.int_(x), dtype=float)

    def recurse(self, data, label, info_d, dic, max_depth=np.inf,
                depth=0, feature_idx=None, is_seq=1, method='id3'):
        outputs = np.unique(label)
        n_features = data.shape[1]
        n_samples = data.shape[0]
        if len(outputs) == 1:
            dic['output'] = outputs[0]
            return
        if len(outputs) == 0:
            dic['output'] = -1
            return
        if depth >= max_depth:
            dic['output'] = collections.Counter(label).most_common(1)[0][0]
            return
        if n_features == 0 or n_samples == 0:
            if len(label) == 0:
                dic['output'] = None
            else:
                dic['output'] = collections.Counter(label).most_common(1)[0][0]
            return
        # count info(status)
        info_ij = []
        statuss = []
        p_i = [np.sum(label == i) / n_samples for i in outputs]
        for i in range(n_features):
            info_ij.append([])
            status = np.unique(data[:, i])
            statuss.append(status)
            for j in status:
                idx = np.where(data[:, i] == j)
                n = len(idx[0])
                p = [np.sum(label[idx] == k) / n for k in outputs]
                if method == 'id3':
                    info_ij[-1].append((n / n_samples, self.get_info(p), j))  # (pj, info_j, j)
                elif method == 'c45':
                    p_split = [p[_] / p_i[_] for _ in range(len(p))]
                    info_ij[-1].append(
                        (n / n_samples, self.get_info(p), j, self.get_info(p_split)))  # (pj, info_j, j, info_split)
                elif method == 'cart':
                    info_ij[-1].append((n / n_samples, self.get_info(p, method='gini'), j))  # (pj, info_j, j)

        # count gain(features)
        gain_i = []
        for i in range(len(info_ij)):
            _ = info_d
            for j in info_ij[i]:
                if method == 'id3':
                    _ -= j[0] * j[1]
                elif method == 'c45':
                    _ = (_ - j[0] * j[1] + 1) / (j[3] + 1)
                elif method == 'cart':
                    _ += j[0] * j[1]
            gain_i.append(_)

        argsort = np.argsort(gain_i)

        if method == 'cart':
            flag = 0
        else:
            flag = -1
        argidx = feature_idx[argsort[flag]]
        dic[argidx] = {}
        if is_seq:
            arg = np.argmax([info_ij[argsort[flag]][i][1] for i in range(len(info_ij[argsort[flag]]))])
            m = info_ij[argsort[flag]][int(arg)][1]
            _ = []
            info = 0
            for i in info_ij[argsort[flag]]:
                if i[1] == m:
                    _.append(i[2])
                    info += i[1]
            mean = np.mean(_)
            dic[argidx][(-np.inf, mean)] = {}
            idx = np.where(data[:, argsort[flag]] <= mean)
            data_ = data[idx]
            label_ = label[idx]
            self.recurse(data_, label_, info, dic[argidx][(-np.inf, mean)],
                         max_depth, depth + 1, feature_idx, is_seq, method=method)

            dic[argidx][(mean, np.inf)] = {}
            idx = np.where(data[:, argsort[flag]] > mean)
            data_ = data[idx]
            label_ = label[idx]
            self.recurse(data_, label_, info, dic[argidx][(mean, np.inf)],
                         max_depth, depth + 1, feature_idx, is_seq, method=method)

        else:
            for i, j in enumerate(statuss[argsort[flag]]):
                dic[argidx][(j,)] = {}
                idx = np.where(data[:, argsort[flag]] == j)
                data_ = data[idx]
                data_ = data_[:, argsort[:flag]]
                label_ = label[idx]
                self.recurse(data_, label_, info_ij[argsort[flag]][i][1], dic[argidx][(j,)],
                             max_depth, depth + 1, feature_idx[argsort[:flag]], is_seq, method=method)

    def get_info(self, p, method='entropy'):
        """
        return info, gain of p
            info = -SUM(p*log2(p)), if p = 0 , p*log2(p) = 0, method = entropy
                 = -SUM(p^2),   method = gini
        """
        info = 0
        if method == 'entropy':
            for i in p:
                if i == 0:
                    continue
                info -= i * np.log2(i)
        elif method == 'gini':
            info = 1
            for i in p:
                info -= i ** 2
        return info

    def predict(self, data):
        data = np.array(data)
        pre = []
        for i in data:
            d = self.dic
            while 'output' not in d:
                for k1, v1 in d.items():  # k1:0, v1:{}
                    for k2, v2 in v1.items():  # k2: ('rainy',) v2:{}
                        if self.is_seq:
                            if k2[0] < i[k1] <= k2[1]:
                                d = v2
                        else:
                            if i[k1] in k2:  # i[k1]: 'rainy'
                                d = v2
            pre.append(d['output'])
        return np.array(pre)

    def show(self, x, y, is_save, save_fn):
        fig, ax = plt.subplots()
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx = np.arange(x_min, x_max, 0.1)
        yy = np.arange(y_min, y_max, 0.1)
        xx, yy = np.meshgrid(xx, yy)
        z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AAAAAA', '#FFFFFF'])
        ax.pcolormesh(xx, yy, z, cmap=cmap_light)
        ax.scatter(x[:, 0], x[:, 1], c=y)
        if is_save:
            fig.savefig(save_fn)
        plt.show()


def create_data():
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
    labels = ['outlook', 'temperature', 'humidity', 'windy', 'play']
    return np.array(datasets), labels


if __name__ == '__main__':
    from sklearn import datasets

    data, _ = create_data()
    x, y = data[:, :-1], data[:, -1:]
    model = DT(x, y, is_seq=0, method='id3')
    print(model.dic)

    x, y = datasets.make_blobs(n_samples=500, n_features=2, centers=5)

    model = DT(x, y, is_seq=1, method='id3', max_depth=10,
               draw=1, is_save=0, save_fn='../img/DT_id3.png')
    # print(model.dic)

    model = DT(x, y, is_seq=1, method='c45', max_depth=10,
               draw=1, is_save=0, save_fn='../img/DT_c45.png')
    # print(model.dic)

    model = DT(x, y, is_seq=1, method='cart', max_depth=10,
               draw=1, is_save=0, save_fn='../img/DT_cart.png')
    # print(model.dic)
