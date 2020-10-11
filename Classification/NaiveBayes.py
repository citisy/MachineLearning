import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


class NB(object):
    def __init__(self, data, label):
        """
        :param data:
        :param label:
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
        self.data = np.array(data, dtype=float)
        self.label = np.array(label)
        self.n_samples = self.data.shape[0]
        self.n_feature = self.data.shape[1]
        self.n_i = len(np.unique(self.data))
        dict_ = collections.Counter(self.label)
        self.dict_ = {}
        for (k, v) in dict_.items():
            self.dict_[k] = {}
            self.dict_[k]['index'] = np.where(self.label == k)
            self.dict_[k]['count'] = v
            self.dict_[k]['py'] = v / self.n_samples
        self.n_features = len(self.dict_)
        self.pole = []
        for i in range(self.n_feature):
            self.pole.append(max(abs(self.data[:, i].min()), abs(self.data[:, i].max())))

    def norm(self, data):
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
        data = np.array(data, dtype=float)
        for i in range(self.n_feature):
            data[:, i] /= self.pole[i] * 0.1

        return data

    def gaussian_predict_(self, data):
        """
        if the gaussian use without trained, use this method
        """
        self.data = self.norm(self.data)
        for k in self.dict_.keys():
            mu = np.mean(self.data[self.dict_[k]['index']], axis=0)
            sigma = np.mat(np.cov(self.data[self.dict_[k]['index']], rowvar=False))
            self.dict_[k]['mu'] = mu
            self.dict_[k]['sigma'] = sigma
        pre = self.gaussian_predict(data)
        return pre

    # 高斯判别
    def gaussian_predict(self, data):
        x = np.array(data)
        n_sample = len(x)
        pre = np.zeros(n_sample, dtype=int)
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
        fig1, ax = plt.subplots()
        fig2, _ = plt.subplots()
        ax2 = Axes3D(fig2)

        for k in self.dict_.keys():
            x = np.arange(self.dict_[k]['mu'][0] - 3, self.dict_[k]['mu'][0] + 3, 0.1)
            y = np.arange(self.dict_[k]['mu'][1] - 3, self.dict_[k]['mu'][1] + 3, 0.1)
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

        # draw interface
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        x = np.arange(x_min, x_max, 0.1)
        y = np.arange(y_min, y_max, 0.1)
        x, y = np.meshgrid(x, y)
        z = self.gaussian_predict(np.c_[x.ravel(), y.ravel()])
        z = z.reshape(x.shape)
        # todo: 分界面呈弧形
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AAAAAA'])
        ax.pcolormesh(x, y, z, cmap=cmap_light)
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.label)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        # fig1.savefig('../img/gaussian_predict2D.png')
        # fig2.savefig('../img/gaussian_predict3D.png')
        plt.show()


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB  # 高斯预测

    x, y = datasets.make_blobs(centers=3)
    model = NB(x, y)
    pre = model.gaussian_predict_(model.norm(x))
    acc = np.sum(pre == y)/len(x)
    print(acc)
    model.show()

    m = GaussianNB().fit(x, y)
    pre = m.predict(x)
    acc = np.sum(pre == y)/len(x)
    print(acc)

    # # 多项式预测
    # from sklearn.naive_bayes import MultinomialNB
    #
    # x = np.random.randint(5, size=(6, 100))
    # y = np.array([1, 2, 3, 4, 5, 6])
    # model = NB(x, y)
    # pre = model.multinomial_predict(x)
    # acc = np.sum(pre == y)/len(x)
    # print(acc)
    # m = MultinomialNB().fit(x, y)   # label >= 0
    # pre = m.predict(x)
    # acc = np.sum(pre == y)/len(x)
    # print(acc)

    # # 伯努利判别
    # from sklearn.naive_bayes import BernoulliNB
    #
    # x = np.random.randint(2, size=(6, 100))
    # y = np.array([1, 2, 3, 4, 5, 6])
    # model = NB(x, y)
    # pre = model.bernoulli_predict(x)
    # acc = np.sum(pre == y)/len(x)
    # print(acc)
    # m = BernoulliNB().fit(x, y)
    # pre = m.predict(x)
    # acc = np.sum(pre == y)/len(x)
    # print(acc)
