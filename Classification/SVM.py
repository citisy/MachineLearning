# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class SVM(object):
    def __init__(self, data, label, c=0.6, tol=0, max_iter=10):
        '''
        :param data:
        :param c:对不在界内的惩罚因子
        :param tol:容忍极限值
        :param itera:最大迭代次数
        '''
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
        self.ecache = np.zeros(len(self.data), dtype=int)
        self.train()

    # 数据归一化
    def norm(self):
        amax = self.data.max()
        self.data /= amax

    def train(self):
        n = len(self.data)
        itera = 0
        while itera < self.max_iter:
            a_change = 0  # a改变的次数
            for i in range(n):
                ei = self.getE(i, i)
                # 不满足kkt条件，确定需要更新的ai
                if not self.is_kkt(ei, i):
                    # 确定aj
                    j, ej = self.getj(i, ei)
                    # 确定上下界
                    l, h = self.getLH(i, j)
                    if l == h:
                        continue
                    # 确定eta
                    eta = self.getEta(i, j)
                    #  如果eta等于0或者小于0 则表明a最优值应该在L或者H上
                    if eta <= 0:
                        continue
                    ai_old = self.a[i]
                    aj_old = self.a[j]
                    # 更新aj
                    self.a[j] += self.label[j] * (ei - ej) / eta
                    if self.a[j] - aj_old < 1e-8:
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
            self.k = -self.w[1] / self.w[0]
            self.draw()
            print(self.w)
            # print(self.label)
        print("train complete!")
        # plt.show()

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
        return True

    def kernel(self, i, j):
        return np.dot(self.data[i], self.data[j])

    def getE(self, ix, iy):
        """
        Ei = ui - yi
        """
        u = self.getu(ix)
        return u - self.label[iy]

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
        return:
            [j, ej]
        """
        self.ecache[i] = i
        max_e = 0
        j = 0
        ej = 0
        flag = 0
        for a in self.ecache:
            if a != 0 and a != i:
                flag = 1
                ea = self.getE(a, a)
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
            ej = self.getE(j, j)
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
        if self.a[i] > 0 and self.a[i] < self.c:
            return b1
        if self.a[j] > 0 and self.a[j] < self.c:
            return b2
        return (b1 + b2) / 2

    def predict(self, i):
        pre = self.getu(i)
        if pre < 0:
            return -1
        else:
            return 1

    def draw(self):
        plt.clf()
        plt.scatter(self.x, self.y, c=self.pre)
        x = [i / 10 for i in range(-10, 10)]
        y = []
        y1 = []
        y2 = []
        for i in x:
            y_ = self.k * i + self.b
            y.append(y_)
            y1.append(y_ + 1)
            y2.append((y_ - 1))
        plt.plot(x, y)
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.draw()
        plt.pause(0.01)


if __name__ == '__main__':
    from sklearn import datasets

    x, y = datasets.make_blobs(centers=2)
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    svm = SVM(x, y)
