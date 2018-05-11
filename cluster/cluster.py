#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class cluster(object):
    def __init__(self, data, k, itera=100, draw=0):
        self.data = np.array(data)
        self.k = k
        self.itera = itera
        self.draw = draw
        self.row = np.shape(self.data)[1]   # 数据维度
        self.col = np.shape(self.data)[0]   # 数据数量
        self.init_cent()
        self.train()


    # 初始化中心点位置
    def init_cent(self):
        self.cent = np.zeros((self.k, self.row))
        for i in range(self.row):
            amax = self.data[:, i].max()
            amin = self.data[:, i].min()
            # 数据归一化
            # self.data = (self.data-amin)/(amax-amin)
            self.data[:, i] /= amax
            # 随机生成中心点落在数据中
            self.cent[:, i] = np.random.random(self.k)*(amax-amin)/amax + amin/amax

    def train(self):
        self.cent_ind = None

    # 求点与中心点的距离
    def get_r(self, x1, x2):
        # 欧氏距离
        return ((x1 - x2) ** 2).sum()
        # 余弦相似度
        # return -np.dot(x1, x2)/np.sqrt((x1**2).sum()*(x2**2).sum())

    # 二维可视化
    def show_(self):
        pass

    def score(self):
        # Calinski-Harabasz score
        # 簇间协方差的迹
        bk = np.trace(np.cov(self.cent, rowvar=False))
        # 簇内协方差的迹
        cent_ = []
        wk = 0
        for i in range(self.k):
            for j in range(self.col):
                if self.cent_ind[j] == i:
                    cent_.append(self.data[j])
            wk += np.trace(np.cov(cent_, rowvar=False))
        return bk*(self.col-self.k)/(wk*(self.k-1))
