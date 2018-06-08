# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import collections


class NB(object):
    def __init__(self, data, label):
        """
        :param data:
        :param label:
        :param rules: data每一个维度的规则
        -----------
        Attributes:
            dict_: dict, shape(n_samples)
                general:
                    'index':
                    'count':
                    'py':
                    'pyx':
                gaussian:
                    'mu':
                    'sigma':
                    'pyx':
                multinomial:
                    'phi':
                    'n_xi':
        """
        self.data = data
        self.label = label
        self.n_samples = len(self.label)
        self.n_i = len(np.unique(self.data))
        dict_ = collections.Counter(self.label)
        self.dict_ = {}
        for (k, v) in dict_.items():
            self.dict_[k] = {}
            self.dict_[k]['index'] = np.where(self.label == k)
            self.dict_[k]['count'] = v
            self.dict_[k]['py'] = v / self.n_samples
        self.n_features = len(self.dict_)

    # 高斯判别
    def gaussian_predict(self, x):
        n_sample = len(x)
        pre = np.zeros(n_sample, dtype=int)
        for k in self.dict_.keys():
            mu = np.mean(self.data[self.dict_[k]['index']], axis=0)
            sigma = np.mat(np.cov(self.data[self.dict_[k]['index']], rowvar=False))
            self.dict_[k]['mu'] = mu
            self.dict_[k]['sigma'] = sigma
        for j in range(n_sample):
            for k in self.dict_.keys():
                delta = x[j] - self.dict_[k]['mu']
                pxy = np.exp(-0.5 * np.matmul(np.matmul(delta.T, self.dict_[k]['sigma'].I), delta)) \
                      / np.sqrt(2 * np.pi * np.linalg.det(self.dict_[k]['sigma']))
                self.dict_[k]['pyx'] = pxy * self.dict_[k]['py']
            # pyx一定大于0，初始赋-1就是无穷小值了
            max = -1
            for k in self.dict_.keys():
                b = self.dict_[k]['pyx']
                if b > max:
                    max = b
                    max_k = k
            pre[j] = max_k
        return pre

    # 多项式判别
    # 加入了拉普拉斯平滑，防止分子为0的情况
    def multinomial_predict(self, x, alpha=1):
        n_sample = len(x)
        n_feature = len(self.data[0])
        pre = np.zeros(n_sample, dtype=int)
        for a in range(n_sample):
            dict_ = collections.Counter(x[a])
            for k in self.dict_.keys():
                self.dict_[k]['phi'] = []
                self.dict_[k]['n_xi'] = []
                # todo: 这种计算每个类别的数量并不严谨，因为并不是每个类别的每一列都是相等的，
                # 但这里传进来的数据是规整的，所以姑且先用这种简易的的做法
                ny = n_feature * self.dict_[k]['count']
                for kx, vx in dict_.items():
                    nyi = np.sum(self.data[self.dict_[k]['index']] == kx)
                    self.dict_[k]['phi'].append((nyi + alpha) / (ny + self.n_i * alpha))
                    self.dict_[k]['n_xi'].append(vx)
                pxy = 1
                for i, j in zip(self.dict_[k]['phi'], self.dict_[k]['n_xi']):
                    pxy *= np.power(i, j)
                self.dict_[k]['pyx'] = pxy * self.dict_[k]['py']
            max = -1
            for k in self.dict_.keys():
                b = self.dict_[k]['pyx']
                if b > max:
                    max = b
                    max_k = k
            pre[a] = max_k
        return pre

    # 伯努利判别，输入值为二值，非真即假
    # 文本分类中的意义：1代表单词有出现过，0代表单词没有出现
    def bernoulli_predict(self, x):
        n_sample = len(x)
        n_feature = len(self.data[0])
        pre = np.zeros(n_sample, dtype=int)
        for a in range(n_sample):
            dict_ = collections.Counter(x[a])
            for k in self.dict_.keys():
                self.dict_[k]['phi'] = []
                self.dict_[k]['n_xi'] = []
                for kx, vx in dict_.items():
                    phi = 1
                    for i in self.dict_[k]['index']:
                        phi *= np.sum(self.data[i] == kx) / n_feature
                    self.dict_[k]['phi'].append(phi)
                    self.dict_[k]['n_xi'].append(vx)
                pxy = 1
                for i, j in zip(self.dict_[k]['phi'], self.dict_[k]['n_xi']):
                    pxy *= np.power(i, j)
                self.dict_[k]['pyx'] = pxy * self.dict_[k]['py']
            max = -1
            for k in self.dict_.keys():
                b = self.dict_[k]['pyx']
                if b > max:
                    max = b
                    max_k = k
            pre[a] = max_k
        return pre

    def show(self):
        from mpl_toolkits.mplot3d import Axes3D
        plt.clf()
        ax = plt.gca()
        fig = plt.figure()
        ax2 = Axes3D(fig)
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.label)
        for k in self.dict_.keys():
            x = np.arange(self.dict_[k]['mu'][0] - 5, self.dict_[k]['mu'][0] + 5, 0.1)
            y = np.arange(self.dict_[k]['mu'][1] - 5, self.dict_[k]['mu'][1] + 5, 0.1)
            x, y = np.meshgrid(x, y)
            d = np.linalg.det(self.dict_[k]['sigma'])
            z = np.exp(
                -0.5 * ((x - self.dict_[k]['mu'][0]) ** 2 + (y - self.dict_[k]['mu'][1]) ** 2) / d ** 2) / np.sqrt(
                2 * np.pi * d)
            # 等高线
            # 二维高斯函数的切面是一个椭圆
            # todo： 二维高斯函数的切面方程：
            cs = ax.contour(x, y, z)
            ax.clabel(cs, inline=1, fontsize=10)
            ax2.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        # todo: 以两个中心点坐标的均值划分分界线是不准确的，还得考虑方差的情况
        # todo：当数据的中心数大于2时，要怎样可是化
        # k = (self.dict_[1]['mu'][1] - self.dict_[0]['mu'][1]) / (self.dict_[1]['mu'][0] - self.dict_[0]['mu'][0])
        # mid = [(self.dict_[1]['mu'][0] + self.dict_[0]['mu'][0]) / 2, (self.dict_[1]['mu'][1] + self.dict_[0]['mu'][1]) / 2]
        # x = np.linspace(self.data[:, 0].min()-1, self.data[:, 0].max()+1)
        # y = -(x - mid[0]) / k + mid[1]
        # ax.plot(x, y)
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets

    # 高斯预测
    from sklearn.naive_bayes import GaussianNB
    # data = datasets.load_iris()
    # x = data.data
    # y = data.target
    # x, y = datasets.make_blobs(centers=2)
    # model = NB(x, y)
    # pre = model.gaussian_predict(x)
    # for i in range(len(x)):
    #     print(pre[i], y[i])
    # model.show()
    # m = GaussianNB().fit(x,y)
    # pre = m.predict(x)
    # for i in range(len(x)):
    #     print(pre[i], y[i])

    # # 多项式预测
    # from sklearn.naive_bayes import MultinomialNB
    #
    # x = np.random.randint(5, size=(6, 100))
    # y = np.array([1, 2, 3, 4, 5, 6])
    # model = NB(x, y)
    # pre = model.multinomial_predict(x)
    # for i in range(6):
    #     print(pre[i], y[i])
    # clf = MultinomialNB().fit(x, y)
    # print(clf.predict(x))

    # # 伯努利判别
    from sklearn.naive_bayes import BernoulliNB

    x = np.random.randint(2, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    clf = BernoulliNB().fit(x, y)
    print(clf.predict(x))
    model = NB(x, y)
    pre = model.bernoulli_predict(x)
    for i in range(6):
        print(pre[i], y[i])
