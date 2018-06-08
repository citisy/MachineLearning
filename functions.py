import numpy as np


class Functions(object):
    def __init__(self):
        self.sigmoid_table = None

    def tanh(self, x):
        # return np.tanh(x)
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def tanh_(self, x):
        return 1 - self.tanh(x) ** 2

    def sigmoid(self, x, k=1, x0=0):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def sigmoid_(self, x):
        return self.sigmoid(x) - self.sigmoid(x) ** 2

    def relu(self, x):
        return max(0, x)

    def relu_(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def softmax(self, x):
        y = []
        sum_exp = np.exp(x).sum()
        for i in x:
            y.append(np.exp(i) / sum_exp)
        return np.array(y)

    def softmax_(self, x, ind):
        x[ind] -= 1
        return x

    def sigmoid_fast(self, x, mins=-6, maxs=6):
        # 缓存(maxs-mins)*100个数
        if self.sigmoid_table == None:
            self.sigmoid_table = []
            for i in range(mins * 100, maxs * 100):
                self.sigmoid_table.append(self.sigmoid(i / 100))
        # 查表输出
        if x <= -6:
            return -1
        elif x >= 6:
            return 1
        else:
            return self.sigmoid_table[int(x * 100)]

    def dropout(self, x, drop_pro):
        for i in range(len(x)):
            if np.random.random > drop_pro:
                x[i] = 0
        return x

    # 协方差矩阵
    def cov(self, x):
        # return np.cov(z, rowvar=False)
        row = np.shape(x)[1]  # 样本维度
        col = np.shape(x)[0]  # 样本数
        covl = np.zeros((row, row))
        for i in range(row):
            xi = x[:, i]
            ximean = np.mean(xi)
            for j in range(row):
                xj = x[:, j]
                xjmean = np.mean(xj)
                covl[i, j] += np.dot((xi - ximean).T, (xj - xjmean)) / (col - 1)
        return covl

    # 矩阵的迹
    def trace(self, x):
        # return np.trace(x)
        if np.shape(x)[0] != np.shape(0)[1]:
            return 'shape exception!'
        tra = 0
        for i in range(np.shape(x)[0]):
            tra += x[i][i]
        return tra

    # 欧氏距离
    def dist(self, x, y):
        return np.sqrt(((x - y) ** 2).sum())

    # 余弦相似度
    def cos_sim(self, x, y):
        return np.dot(x, y) / np.sqrt((x ** 2).sum() * (y ** 2).sum())

    def make_blobs(self, n_centers=2, mean=None, cov=None, n_sample=None):
        mean = mean or [[2, 3], [7, 8]]
        cov = cov or [[[1, 0], [0, 2]],
                      [[1, 0], [0, 2]]]
        n_sample = n_sample or [500, 500]
        x = []
        y = []
        for i in range(n_centers):
            x.append(np.random.multivariate_normal(mean[i], cov[i], n_sample[i]))
            y.extend([i] * n_sample[i])
        return x, np.array(y)


if __name__ == '__main__':
    f = Functions()
    # z = np.array([[1, 2, 3, 4], [3, 4, 1, 2], [2, 3, 1, 4]])
    # print(f.dist(z[:, 0], z[:, 1]))
    x, y = f.make_blobs()
    print(y)
