# -*- coding: utf-8 -*-

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)


class SVM(object):
    def __init__(self, data, label, c=1.0, tol=1e-3, max_iter=10, kernel='liner', draw=0,
                 gamma=None, r=None, d=None):
        """
        Parameters:
            c:
                对不在界内的惩罚因子
            tol:
                容忍极限值
            itera:
                最大迭代次数
            kernel:
                'liner' -> x * x'
                'rbf' -> exp(-gamma(||x - x'||^2))
                'polynomial' -> gamma(x * x' + r)^d
            multi_class:
                0 -> normal classification
                1 -> 1vr classification
                2 -> 1v1 classification
        """
        self.data = np.array(data)
        self.label = np.array(label)  # 数据标签，分为-1和+1
        self.c = c
        self.tol = tol
        self.max_iter = max_iter
        self.kernel = kernel
        self.draw = draw
        self.gamma = gamma or 0.1
        self.r = r or 1
        self.d = d or 2
        self.n_sample = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self._class = np.unique(self.label)
        self.n_class = len(self._class)

        self.pole = []
        for i in range(self.n_features):
            self.pole.append(max(abs(self.data[:, i].min()), abs(self.data[:, i].max())))
        self.data = self.norm(self.data)

        if self.draw:
            self.ims = []
            self.col = math.ceil(np.sqrt(self.n_features / 2))
            self.row = math.ceil(self.n_features / 2 / self.col)
            self.fig, self.ax = plt.subplots(ncols=self.col, nrows=self.row, squeeze=False)
            self.ax[0][0].set_ylim(self.label.min() * 1.2, self.label.max() * 1.2)
            self.fig.set_tight_layout(True)

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
        for i in range(self.n_features):
            data[:, i] /= self.pole[i] * 1.2

        return data

    def train(self):
        n = self.n_sample
        pre = np.zeros(self.n_sample)
        itera = 0
        while itera < self.max_iter:
            a_change = 0  # a改变的次数
            for i in range(n):
                ei = self.getE(i)
                # 满足kkt -> 已经得到最优解，不需要进行操作
                # 不满足kkt条件 -> 还没得到最优解，进行优化
                if not self.is_kkt(ei, i):
                    # 确定aj
                    j, ej = self.getj(i, ei)
                    # 确定上下界
                    l, h = self.getLH(i, j)
                    # l等于h时，a的值在l或h上
                    # 即|aj-ai|=c
                    if l == h:
                        continue
                    # 确定eta
                    eta = self.getEta(i, j)
                    #  如果eta等于0或者小于0 则表明a最优值应该在L或者H上
                    ai_old = self.__a[i]
                    aj_old = self.__a[j]
                    # 更新aj
                    self.__a[j] += self.__label[j] * (ei - ej) / eta
                    if np.abs(self.__a[j] - aj_old) <= 1e-4:
                        continue
                    # 下界是L 也就是截距,小于L时为L
                    # 上界是H 也就是最大值,大于H时为H
                    # L <= aj <= H
                    self.__a[j] = min(self.__a[j], h)
                    self.__a[j] = max(self.__a[j], l)
                    self.__a[i] += self.__label[i] * self.__label[j] * (aj_old - self.__a[j])
                    # j是随机挑选的情况
                    if 0 < self.__a[i] < self.c:
                        self.__ecache[i] = i
                    if 0 < self.__a[j] < self.c:
                        self.__ecache[j] = j

                    # 更新b
                    self.__b = self.getb(i, j, ei, ej, ai_old, aj_old)
                    a_change += 1
            if a_change == 0:
                itera += 1
            else:
                itera = 0
            self.__w = self.getW()
            for i in range(self.n_sample):
                pre[i] = self.predict(i, self.__w)
            if self.draw:
                self.show(self.__data, pre, [self.__w], [self.__b])

    def svc(self):
        stime = time.time()
        self.__b = 0
        self.__w = np.zeros(self.n_features)
        self.__a = np.zeros(self.n_sample)  # 拉格朗日乘子
        self.__ecache = np.zeros(self.n_sample, dtype=int) - 1
        self.__data = self.data.copy()
        self.__label = self.label.copy()
        self.__label[np.where(self.__label == self._class[0])] = -1
        self.__label[np.where(self.__label != -1)] = 1
        self.train()
        etime = time.time()
        print('train completed! time: %s' % str(etime - stime))
        if self.draw:
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=3000 / len(self.ims), blit=True,
                                            repeat_delay=0, repeat=True)
            # ani.save('../img/SVM_svc.gif', writer='imagemagick')
            plt.show()

    def _1vr(self):
        """
        1对多：
            开销少，但有时候分类效果不太好
            例如，3个位于同一平行线上的类，中间的那个类分类效果就不好
        :return:
        """
        stime = time.time()

        self.b = np.zeros(self.n_class)
        self.w = np.zeros((self.n_class, self.n_features))
        self.a = np.zeros((self.n_class, self.n_sample))  # 拉格朗日乘子
        self.ecache = np.zeros((self.n_class, self.n_sample), dtype=int) - 1
        self.label_cache = np.zeros((self.n_class, self.n_sample))
        self.__data = self.data.copy()
        for i in range(self.n_class):
            self.__b = 0
            self.__w = np.zeros(self.n_features)
            self.__a = np.zeros(self.n_sample)  # 拉格朗日乘子
            self.__ecache = np.zeros(self.n_sample, dtype=int) - 1
            self.__label = self.label.copy()
            self.__label[np.where(self.__label == self._class[i])] = -1
            self.__label[np.where(self.__label != -1)] = 1
            self.train()
            self.b[i] = self.__b
            self.w[i] = self.__w
            self.a[i] = self.__a
            self.ecache[i] = self.__ecache
            self.label_cache[i] = self.__label

        etime = time.time()
        print('train completed! time: %s' % str(etime - stime))

        if self.draw:
            pre = self.predict_1vr(self.data)

            for _ in range(len(self.ims)):
                self.show(self.data, pre, self.w, self.b, 0)

            ani = animation.ArtistAnimation(self.fig, self.ims, interval=3000 / len(self.ims), blit=True,
                                            repeat_delay=0, repeat=True)
            # ani.save('../img/SVM_1vr.gif', writer='imagemagick')
            plt.show()

    def _1v1(self):
        stime = time.time()

        self.b = []
        self.w = []
        self.a = []  # 拉格朗日乘子
        self.ecache = []
        self.label_cache = []
        self.data_cache = []
        a = 0
        for i in range(self.n_class):
            for j in range(i + 1, self.n_class):
                zeros = np.where(self.label == self._class[i])
                ones = np.where(self.label == self._class[j])
                self.__label = self.label[np.concatenate(zeros + ones)]
                self.__data = self.data[np.concatenate(zeros + ones)]
                self.__label[np.where(self.__label == self._class[i])] = -1
                self.__label[np.where(self.__label != -1)] = 1
                self.n_sample = len(self.__label)
                self.__b = 0
                self.__w = np.zeros(self.n_features)
                self.__a = np.zeros(self.n_sample)  # 拉格朗日乘子
                self.__ecache = np.zeros(self.n_sample, dtype=int) - 1
                self.train()
                self.b.append(self.__b)
                self.w.append(self.__w)
                self.a.append(self.__a)
                self.ecache.append(self.__ecache)
                self.label_cache.append(self.__label)
                self.data_cache.append(self.__data)
            a += 1

        etime = time.time()
        print('train completed! time: %s' % str(etime - stime))
        if self.draw:
            pre = self.predict_1v1(self.data)

            for _ in range(len(self.ims)):
                self.show(self.data, pre, self.w, self.b, 0)

            ani = animation.ArtistAnimation(self.fig, self.ims, interval=3000 / len(self.ims), blit=True,
                                            repeat_delay=0, repeat=True)
            # ani.save('../img/SVM_1v1.gif', writer='imagemagick')
            plt.show()

    # 判断是否符合kkt条件
    def is_kkt(self, e, i):
        """
        满足kkt的条件：（0 <= alpha <= c）
            yi*ui >= 1 and alpha == 0 (正确分类)
            yi*ui == 1 and 0<alpha < C (在边界上的支持向量)
            yi*ui <= 1 and alpha == C (在边界之间)
        又：
            ei = ui - yi
            ri = yi * ei = yi * ui -yi^2
        不满足kkt的情况为：
            ri < 0 and alpha < c    （距离小于0，alpha应该等于c，不满足时取下边界）
            ri > 0 and alpha > 0    （距离大于0，alpha应该等于0， 不满足时取上边界）
        return:
            满足 -> True
            不满足 -> False
        """
        if self.__label[i] * e < -self.tol and self.__a[i] < self.c:
            return False
        if self.__label[i] * e > self.tol and self.__a[i] > 0:
            return False
        self.__ecache[i] = -1
        return True

    def __kernel(self, i, j):
        if self.kernel == 'liner':
            return np.matmul(self.__data[i], self.__data[j].T)
        if self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(self.__data[i] - self.__data[j]))
        if self.kernel == 'polynomial':
            return self.gamma(np.matmul(self.__data, self.__data[j].T) + self.r) ^ self.d

    def getE(self, i):
        """
        Ei = ui - yi
        """
        u = self.getu(i)
        return u - self.__label[i]

    # 目标值
    def getu(self, i, w=None):
        """
        ui -> pre
            ui = w *　xi * k + b
        """
        if w is None:
            w = self.getW()
        u = np.dot(w, self.__data[i]) + self.__b
        return u

    def getW(self):
        """
        w: 平面的法向量
        二维为例：
            w = [w1, w2]
            x = [x1, x2]
            平面(二维为直线)簇方程：g(x) = w * x + b -> w1x1 + w2x2 + b
            中心直线方程：g(x) = 0
        :return:
        """
        w = 0
        for i in range(self.n_sample):
            w += self.__a[i] * self.__label[i] * self.__data[i]
        return w

    def getj(self, i, ei):
        """
        启发式遍历：
            对于上一次不满足kkt的点，下一次很大概率也不满足。
            故我们只需要遍历上一次不满足的点就可以了。
            在这些点中，|ej-ei|的值最大就是我们得到的j了
            其实，ej是随机选取都是没有问题的
        return:
            [j, ej]
        """
        self.__ecache[i] = i
        max_e = 0
        j = 0
        ej = 0
        flag = 0
        for a in self.__ecache:
            if a != -1 and a != i:
                flag = 1
                ea = self.getE(a)
                delta_e = np.abs(ea - ei)
                if delta_e > max_e:
                    max_e = delta_e
                    j = a
                    ej = ea
        # 没有适合的j，随机选取一个
        if not flag:
            j = np.random.randint(self.n_sample)
            while j == i:
                j = np.random.randint(self.n_sample)
            ej = self.getE(j)
        return j, ej

    def getLH(self, i, j):
        if self.__label[i] == self.__label[j]:
            l = max(0.0, self.__a[j] + self.__a[i] - self.c)
            h = min(self.c, self.__a[j] + self.__a[i])
        else:
            l = max(0.0, self.__a[j] - self.__a[i])
            h = min(self.c, self.c + self.__a[j] - self.__a[i])
        return l, h

    def getEta(self, i, j):
        """

        """
        eta = self.__kernel(i, i) + self.__kernel(j, j) - 2 * self.__kernel(i, j)
        return eta

    def getb(self, i, j, ei, ej, ai_old, aj_old):
        b1 = (self.__b - ei - self.__label[i] * (self.__a[i] - ai_old) * self.__kernel(i, i)
              - self.__label[j] * (self.__a[j] - aj_old) * self.__kernel(i, j))
        b2 = (self.__b - ej - self.__label[i] * (self.__a[i] - ai_old) * self.__kernel(i, j)
              - self.__label[j] * (self.__a[j] - aj_old) * self.__kernel(j, j))
        if 0 < self.__a[i] < self.c:
            return b1
        if 0 < self.__a[j] < self.c:
            return b2
        # 貌似到不了这一步，至少一定存在0<aj<c
        return (b1 + b2) / 2

    def predict(self, i, w=None):
        pre = self.getu(i, w)
        if pre < 0:
            # if pre > -1:
            if self.__a[i] != 0:
                return -2
            return -1
        else:
            # if pre < 1:
            if self.__a[i] != 0:
                return 2
            return 1

    def predict_1vr(self, data):
        data = np.array(data)
        n_sample = data.shape[0]
        pre = np.zeros(n_sample)
        for i in range(n_sample):  # predict
            p = []
            for j in range(self.n_class):
                b = self.b[j]
                w = self.w[j]
                u = np.dot(w, data[i]) + b
                p.append(u)
            # 判断依据：1对多中的‘1’被划分为‘-1’类，故其判断值小于0，故选择预测值最小为预测的类
            pre[i] = self._class[np.argmin(p)]
        return pre

    def predict_1v1(self, data):
        data = np.array(data)
        n_sample = data.shape[0]
        k = self.n_class * (self.n_class - 1) // 2
        pre = np.zeros(n_sample)
        for i in range(n_sample):
            p = np.zeros(self.n_class, dtype=int)
            for j in range(k):
                b = self.b[j]
                w = self.w[j]
                x = 0
                a = self.n_class - 1
                while j - a >= 0:
                    j = j - a
                    a -= 1
                    x += 1
                u = np.dot(w, data[i]) + b
                if u > 0:
                    p[x + j + 1] += 1
                else:
                    p[x] += 1
            pre[i] = self._class[np.argmax(p)]
        return pre

    def show(self, data, pre, w, b, draw_support_line=1):
        im = []
        ax = self.ax[0][0]
        ax.set_xlim(
            [-1.2, 1.2])  # data will be normalized between [-1, 1], so set the axis between [-1.2, 1.2] is enough
        ax.set_ylim([-1.2, 1.2])
        x = np.linspace(-1, 1)
        sca = ax.scatter(data[:, 0], data[:, 1], c=pre)
        im.append(sca)
        for i in range(len(w)):
            # draw line base on w, b, base line function: y = -(w1*x+b)/w2
            y = -(w[i][0] * x + b[i]) / w[i][1]
            line1, = ax.plot(x, y, c='r')
            im.append(line1)
            if draw_support_line:
                y1 = -(w[i][0] * x + b[i] + 1) / w[i][1]
                y2 = -(w[i][0] * x + b[i] - 1) / w[i][1]
                line2, = ax.plot(x, y1, c='b')
                line3, = ax.plot(x, y2, c='b')
                im.append(line2)
                im.append(line3)
        self.ims.append(im)


def sklearn_pre(x, y):
    from sklearn import svm

    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)
    w = clf.coef_[0]
    k = -w[0] / w[1]
    b = clf.intercept_[0]
    pre = clf.support_
    y[pre] = -2
    xx = np.linspace(-1, 1)
    yy = k * xx - b / w[1]
    yy1 = k * xx - (b + 1) / w[1]
    yy2 = k * xx - (b - 1) / w[1]
    plt.figure()
    ax = plt.gca()
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.scatter(x[:, 0], x[:, 1], c=y)
    ax.plot(xx, yy)
    ax.plot(xx, yy1)
    ax.plot(xx, yy2)
    plt.show()


if __name__ == '__main__':
    from sklearn import datasets

    x, y = datasets.make_blobs(centers=3, random_state=23)
    model = SVM(x, y, c=0.5, draw=1, kernel='liner')
    # model.svc()
    # model._1vr()
    model._1v1()
    # sklearn_pre(model.data, model.label)
