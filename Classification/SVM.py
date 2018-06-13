# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class SVM(object):
    def __init__(self, data, label, c=1.0, tol=1e-3, max_iter=10, kernel='liner', show=0,
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
        self.data = data
        self.label = label  # 数据标签，分为-1和+1
        self.c = c
        self.tol = tol
        self.max_iter = max_iter
        self.kernel = kernel
        self.show = show
        self.gamma = gamma or 0.1
        self.r = r or 1
        self.d = d or 2
        self.n_sample = np.shape(self.data)[0]
        self.n_feature = np.shape(self.data)[1]
        self._class = np.unique(self.label)
        self.n_class = len(self._class)

        self.pre = np.zeros(self.n_sample)
        self.norm()
        # self.train()

    # 数据归一化
    def norm(self):
        for i in range(self.n_feature):
            amax = np.abs(self.data[:, i]).max()
            amin = np.abs(self.data[:, i]).min()
            self.data[:, i] /= amax
            # todo: amax不能取绝对值
            # self.__data[:, i] = 2 * (self.__data[:, i] - amin) / (amax - amin) - 1

    def train(self):
        n = self.n_sample
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
                    # #  如果eta等于0或者小于0 则表明a最优值应该在L或者H上
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
                    if self.__a[i] > 0 and self.__a[i] < self.c:
                        self.__ecache[i] = i
                    if self.__a[j] > 0 and self.__a[j] < self.c:
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
                self.pre[i] = self.predict(i, self.__w)
            if self.show:
                self.draw()
        print("train complete!")

    def svc(self):
        self.__b = 0
        self.__w = np.zeros(self.n_feature)
        self.__a = np.zeros(self.n_sample)  # 拉格朗日乘子
        self.__ecache = np.zeros(self.n_sample, dtype=int) - 1
        self.__data = self.data.copy()
        self.__label = self.label.copy()
        self.__label[np.where(self.__label == self._class[0])] = -1
        self.__label[np.where(self.__label != -1)] = 1
        self.train()

    def _1vr(self):
        """
        1对多：
            开销少，但有时候分类效果不太好
            例如，3个位于同一平行线上的类，中间的那个类分类效果就不好
        :return:
        """
        self.b = np.zeros(self.n_class)
        self.w = np.zeros((self.n_class, self.n_feature))
        self.a = np.zeros((self.n_class, self.n_sample))  # 拉格朗日乘子
        self.ecache = np.zeros((self.n_class, self.n_sample), dtype=int) - 1
        self.label_cache = np.zeros((self.n_class, self.n_sample))
        self.__data = self.data.copy()
        for i in range(self.n_class):
            self.__b = 0
            self.__w = np.zeros(self.n_feature)
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
        for i in range(self.n_sample):
            p = []
            for j in range(self.n_class):
                self.__b = self.b[j]
                self.__w = self.w[j]
                self.__a = self.a[j]
                self.__label = self.label_cache[j]
                u = np.dot(self.__w, self.data[i]) + self.__b
                p.append(u)
            # 判断依据：1对多中的‘1’被划分为‘-1’类，故其判断值小于0，故选择预测值最小为预测的类
            self.pre[i] = self._class[np.argmin(p)]
        self.draw()

    def _1v1(self):
        k = self.n_class * (self.n_class - 1) // 2
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
                self.__w = np.zeros(self.n_feature)
                self.__a = np.zeros(self.n_sample)  # 拉格朗日乘子
                self.__ecache = np.zeros(self.n_sample, dtype=int) - 1
                self.pre = np.zeros(self.n_sample)
                self.train()
                self.b.append(self.__b)
                self.w.append(self.__w)
                self.a.append(self.__a)
                self.ecache.append(self.__ecache)
                self.label_cache.append(self.__label)
                self.data_cache.append(self.__data)
            a += 1
        self.n_sample = np.shape(self.data)[0]
        self.pre = np.zeros(self.n_sample)
        for i in range(self.n_sample):
            p = np.zeros(self.n_class, dtype=int)
            for j in range(k):
                self.__b = self.b[j]
                self.__w = self.w[j]
                x = 0
                a = self.n_class - 1
                while j - a >= 0:
                    j = j - a
                    a -= 1
                    x += 1
                u = np.dot(self.__w, self.data[i]) + self.__b
                if u > 0:
                    p[x + j + 1] += 1
                else:
                    p[x] += 1
            self.pre[i] = self._class[np.argmax(p)]
        self.__data = self.data.copy()
        self.draw()

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
        b1 = self.__b - ei - self.__label[i] * (self.__a[i] - ai_old) * self.__kernel(i, i) \
             - self.__label[j] * (self.__a[j] - aj_old) * self.__kernel(i, j)
        b2 = self.__b - ej - self.__label[i] * (self.__a[i] - ai_old) * self.__kernel(i, j) \
             - self.__label[j] * (self.__a[j] - aj_old) * self.__kernel(j, j)
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

    def draw(self):
        plt.clf()
        ax = plt.gca()
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.scatter(self.__data[:, 0], self.__data[:, 1], c=self.pre)
        x = np.linspace(-1, 1)
        # 整合得y = -(w1*x+b)/w2
        y = -(self.__w[0] * x + self.__b) / self.__w[1]
        y1 = -(self.__w[0] * x + self.__b + 1) / self.__w[1]
        y2 = -(self.__w[0] * x + self.__b - 1) / self.__w[1]
        ax.plot(x, y)
        ax.plot(x, y1)
        ax.plot(x, y2)
        plt.draw()
        plt.pause(0.01)


def sklearn_pre(x, y):
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
    from sklearn import svm

    x, y = datasets.make_blobs(centers=2)
    model = SVM(x, y, c=0.5, show=1, kernel='liner')
    model.svc()
    sklearn_pre(model.data, model.label)
