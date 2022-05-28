import numpy as np
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils import Painter

linear = 0
rbf = 1
polynomial = 2


class MyPainter(Painter):
    def img_collections(self, data, label, w, bias, *args, **kwargs):
        for a, b, i, im in self.draw_ani(data, label, w, bias, *args, **kwargs):
            x_min, x_max = data[:, i].min() - 1, data[:, i].max() + 1
            xx = np.arange(x_min, x_max)

            y1 = -(w[0] * xx + bias + 1) / w[1]
            y2 = -(w[0] * xx + bias - 1) / w[1]

            line2, = self.ani_ax[a][b].plot(xx, y1, c='b')
            line3, = self.ani_ax[a][b].plot(xx, y2, c='b')
            im.append(line2)
            im.append(line3)


class SVM(object):
    def __init__(self, kernel=linear, k_args=None, n_features=None,
                 show_img=False, show_ani=False, painter=None):
        self.kernel = kernel
        self.k_args = k_args
        self.show_img = show_img
        self.show_ani = show_ani

        if self.show_img or self.show_ani:
            self.painter = painter or MyPainter(n_features)
            self.painter.beautify()

            if not painter and self.show_img:
                self.painter.init_pic()

            if not painter and self.show_ani:
                self.painter.init_ani()

    def fit(self, data, label, c=1.0, tol=1e-3, max_iter=10, img_save_path=None, ani_save_path=None):
        """
        Parameters:
            data:
                [n_samples, n_features]
            label:
                -1 or +1
            c:
                对不在界内的惩罚因子
            tol:
                容忍极限值
            max_iter:
                最大迭代次数
        """
        data = np.array(data, dtype=float)
        label = np.array(label, dtype=float)

        n_samples = data.shape[0]
        self.a = np.zeros(n_samples)
        self.b = .0
        ecache = np.zeros_like(label)

        for _ in tqdm(range(max_iter)):
            change = False

            for i in range(n_samples):
                ei = self.getE(i, data, label)

                # 满足kkt -> 已经得到最优解，不需要进行操作
                # 不满足kkt条件 -> 还没得到最优解，进行优化
                if not self.is_kkt(label[i], ei, self.a[i], tol, c):
                    j, ej = self.getj(ei, ecache, data, label)  # 确定aj

                    while j == i:  # 如果取的是同一个点，则随机挑选一个点
                        j = np.random.randint(n_samples)
                        ej = self.getE(j, data, label)

                    l, h = self.getLH(i, j, label, c)  # 确定上下界

                    # l等于h时，a的值在l或h上
                    # 即|aj-ai|=c
                    if l == h:
                        continue

                    eta = self.getEta(data[i], data[j])  # 确定eta

                    #  如果eta等于0或者小于0 则表明a最优值应该在L或者H上
                    ai_old = self.a[i]
                    aj_old = self.a[j]

                    self.a[j] += label[j] * (ei - ej) / eta  # 更新aj

                    # 下界是L 也就是截距,小于L时为L
                    # 上界是H 也就是最大值,大于H时为H
                    # L <= aj <= H
                    self.a[j] = min(self.a[j], h)
                    self.a[j] = max(self.a[j], l)
                    self.a[i] += label[i] * label[j] * (aj_old - self.a[j])

                    if np.abs(self.a[j] - aj_old) <= 1e-4:  # 改变量过少，则认为参数未改变
                        continue

                    # 只缓存分类正确的向量
                    # 如果本次优化分类仍不正确，下次优化很可能继续选到该点继续进行优化
                    # 为了节省时间，这里不用再次计算ei和ej，因为这里只是大概地暂存一下
                    # 而且每遍历完一次样本集，都会重新更新一次缓存库
                    if self.is_kkt(label[i], ei, self.a[i], tol, c):
                        ecache[i] = ei
                    if self.is_kkt(label[j], ej, self.a[j], tol, c):
                        ecache[j] = ej

                    self.b = self.getb(i, j, ei, ej, ai_old, aj_old, c, data, label)  # 更新b

                    change = True

                    if self.show_ani:
                        self.w = self.getW(data, label)
                        self.painter.img_collections(data, label, self.w, self.b)

            if not change:  # 如果参数都没发生改变，说明迭代已经结束
                break

            for i in range(n_samples):  # 每次迭代更新后更新一次缓存表
                ecache[i] = self.getE(i, data, label)

        self.w = self.getW(data, label)

        if self.show_ani:
            self.painter.show_ani(ani_save_path)

        if self.show_img:
            self.painter.show_pic(data, label, self.predict, img_save_path)
            self.painter.show()

    # 判断是否符合kkt条件
    def is_kkt(self, yi, ei, ai, tol, c):
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
        if yi * ei < tol and ai < c:
            return False
        if yi * ei > tol and ai > 0:
            return False
        return True

    def K(self, xi, xj):
        """核函数
        liner -> x * x'
        rbf -> exp(-gamma(||x - x'||^2))
        polynomial -> gamma(x * x' + r)^d
        """
        if self.kernel == linear:
            return np.matmul(xi, xj.T)
        if self.kernel == rbf:
            gamma = self.k_args[0]
            return np.exp(-gamma * np.linalg.norm(xi - xj))
        if self.kernel == polynomial:
            gamma, r, d = self.k_args[0:3]
            return gamma(np.matmul(xi, xj.T) + r) ^ d

    def getE(self, i, data, label):
        """Ei = ui - yi"""
        return self.getu(data[i], data, label) - label[i]

    # 目标值
    def getu(self, xi, data, label):
        """
        ui -> pre
            ui = w *　xi * k + b
        """
        n_samples = data.shape[0]
        u = 0
        for j in range(n_samples):
            u += self.a[j] * label[j] * self.K(data[j], xi)
        return u + self.b

    def getW(self, data, label):
        """
        w: 平面的法向量
        二维为例：
            w = [w1, w2]
            x = [x1, x2]
            平面(二维为直线)簇方程：g(x) = w * x + b -> w1x1 + w2x2 + b
            中心直线方程：g(x) = 0
        """
        return data.T @ (label * self.a)

    def getj(self, ei, ecache, data, label):
        """
        启发式遍历：
            对于上一次不满足kkt的点，下一次很大概率也不满足。
            故我们只需要遍历上一次不满足的点就可以了。
            在这些点中，|ej-ei|的值最大就是我们得到的j了
            其实，ej是随机选取都是没有问题的
        return:
            [j, ej]
        """
        if ei < 0:
            j = np.argmax(ecache)
        else:
            j = np.argmin(ecache)

        ej = self.getE(j, data, label)

        return j, ej

    def getLH(self, i, j, label, c):
        if label[i] == label[j]:
            l = max(0.0, self.a[j] + self.a[i] - c)
            h = min(c, self.a[j] + self.a[i])
        else:
            l = max(0.0, self.a[j] - self.a[i])
            h = min(c, c + self.a[j] - self.a[i])
        return l, h

    def getEta(self, xi, xj):
        """eta = Kii + Kjj - 2Kij"""
        eta = self.K(xi, xi) + self.K(xj, xj) - 2 * self.K(xi, xj)
        return eta

    def getb(self, i, j, ei, ej, ai_old, aj_old, c, data, label):
        b1 = (self.b - ei - label[i] * (self.a[i] - ai_old) * self.K(data[i], data[i])
              - label[j] * (self.a[j] - aj_old) * self.K(data[i], data[j]))
        b2 = (self.b - ej - label[i] * (self.a[i] - ai_old) * self.K(data[i], data[j])
              - label[j] * (self.a[j] - aj_old) * self.K(data[j], data[j]))

        if 0 < self.a[i] < c:
            return b1
        if 0 < self.a[j] < c:
            return b2

        return (b1 + b2) / 2

    def predict(self, x):
        return np.int_(np.sign(self.w @ x.T + self.b).reshape(-1))


def sample_test():
    x, y = datasets.make_blobs(centers=2, n_samples=200)
    y[y == 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = SVM(n_features=x.shape[1], show_img=True, show_ani=True)
    model.fit(x_train, y_train,
              # img_save_path='../img/SVM.png',
              # ani_save_path='../img/SVM.mp4',
              )

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.975"""


def real_data_test():
    from MathMethods.Scaler import scaler
    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target

    x, _, _ = scaler.min_max(x)
    y[y == 0] = -1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = SVM()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.9736842105263158"""


def sklearn_test():
    from sklearn.svm import LinearSVC

    dataset = datasets.load_breast_cancer()

    x, y = dataset.data, dataset.target
    y[y == 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LinearSVC()
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print('acc:', np.sum(y_test == pred) / len(y_test))
    """acc: 0.7105263157894737"""


if __name__ == '__main__':
    sample_test()
    # real_data_test()
    # sklearn_test()
