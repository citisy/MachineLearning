# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class SVM(object):
    def __init__(self, data, label, c=1.0, tol=1e-3, max_iter=10):
        """
        :param data:
        :param c:对不在界内的惩罚因子
        :param tol:容忍极限值
        :param itera:最大迭代次数
        """
        self.data = data
        self.norm()
        self.label = label  # 数据标签，分为-1和+1
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.b = 0
        self.w = np.zeros(len(self.data))
        self.c = c
        self.tol = tol
        self.max_iter = max_iter
        self.a = np.zeros(len(self.data))  # 拉格朗日乘子
        self.pre = np.zeros(len(self.data))
        self.ecache = np.zeros(len(self.data), dtype=int) - 1
        self.train()

    # 数据归一化
    def norm(self):
        for i in range(len(self.data[0])):
            amax = np.abs(self.data[:, i]).max()
            amin = np.abs(self.data[:, i]).min()
            self.data[:, i] /= amax
            # todo: amax不能取绝对值
            # self.data[:, i] = 2 * (self.data[:, i] - amin) / (amax - amin) - 1

    def train(self):
        n = len(self.data)
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
                    ai_old = self.a[i]
                    aj_old = self.a[j]
                    # 更新aj
                    self.a[j] += self.label[j] * (ei - ej) / eta
                    if np.abs(self.a[j] - aj_old) <= 1e-4:
                        continue
                    # 下界是L 也就是截距,小于L时为L
                    # 上界是H 也就是最大值,大于H时为H
                    # L <= aj <= H
                    self.a[j] = min(self.a[j], h)
                    self.a[j] = max(self.a[j], l)
                    self.a[i] += self.label[i] * self.label[j] * (aj_old - self.a[j])
                    # j是随机挑选的情况
                    if self.a[i] > 0 and self.a[i] < self.c:
                        self.ecache[i] = i
                    if self.a[j] > 0 and self.a[j] < self.c:
                        self.ecache[j] = j

                    # 更新b
                    self.b = self.getb(i, j, ei, ej, ai_old, aj_old)
                    a_change += 1
            if a_change == 0:
                itera += 1
            else:
                itera = 0
            for i in range(len(self.data)):
                self.pre[i] = self.predict(i)
            self.w = self.getW()
            self.draw()
        print("train complete!")

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
        if self.label[i] * e < -self.tol and self.a[i] < self.c:
            return False
        if self.label[i] * e > self.tol and self.a[i] > 0:
            return False
        self.ecache[i] = -1
        return True

    def kernel(self, i, j):
        return np.dot(self.data[i], self.data[j])

    def getE(self, i):
        """
        Ei = ui - yi
        """
        u = self.getu(i)
        return u - self.label[i]

    # 目标值
    def getu(self, j):
        """
        ui -> pre
            ui = w *　xi * k + b
        """
        w = self.getW()
        u = np.dot(w, self.data[j]) + self.b
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
        for i in range(len(self.data)):
            w += self.a[i] * self.label[i] * self.data[i]
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
        self.ecache[i] = i
        max_e = 0
        j = 0
        ej = 0
        flag = 0
        for a in self.ecache:
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
            j = np.random.randint(len(self.data))
            while j == i:
                j = np.random.randint(len(self.data))
            ej = self.getE(j)
        return j, ej

    def getLH(self, i, j):
        if self.label[i] == self.label[j]:
            l = max(0.0, self.a[j] + self.a[i] - self.c)
            h = min(self.c, self.a[j] + self.a[i])
        else:
            l = max(0.0, self.a[j] - self.a[i])
            h = min(self.c, self.c + self.a[j] - self.a[i])
        return l, h

    def getEta(self, i, j):
        """

        """
        eta = self.kernel(i, i) + self.kernel(j, j) - 2 * self.kernel(i, j)
        return eta

    def getb(self, i, j, ei, ej, ai_old, aj_old):
        b1 = self.b - ei - self.label[i] * (self.a[i] - ai_old) * self.kernel(i, i) \
             - self.label[j] * (self.a[j] - aj_old) * self.kernel(i, j)
        b2 = self.b - ej - self.label[i] * (self.a[i] - ai_old) * self.kernel(i, j) \
             - self.label[j] * (self.a[j] - aj_old) * self.kernel(j, j)
        if 0 < self.a[i] < self.c:
            return b1
        if 0 < self.a[j] < self.c:
            return b2
        # 貌似到不了这一步，至少一定存在0<aj<c
        return (b1 + b2) / 2

    def predict(self, i):
        pre = self.getu(i)
        if pre < 0:
            # if pre > -1:
            if self.a[i] != 0:
                return -2
            return -1
        else:
            # if pre < 1:
            if self.a[i] != 0:
                return 2
            return 1

    def draw(self):
        plt.clf()
        ax = plt.gca()
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.scatter(self.x, self.y, c=self.pre)
        x = np.linspace(-1,1)
        # 整合得y = -(w1*x+b)/w2
        y = -(self.w[0] * x + self.b) / self.w[1]
        y1 = -(self.w[0] * x + self.b + 1) / self.w[1]
        y2 = -(self.w[0] * x + self.b - 1) / self.w[1]
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
    y[pre] = 0
    xx = np.linspace(-1, 1)
    yy = k * xx - b / w[1]
    yy1 = k * xx - (b+1) / w[1]
    yy2 = k * xx - (b-1) / w[1]
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
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    model = SVM(x, y, c=0.5)
    sklearn_pre(model.data, y)
