# -*- coding: utf-8 -*-


import collections
import numpy as np
import jieba
import gensim


class Word2vec(object):
    def __init__(self, train_file, window=5, min_reduce=1,
                 layer1_size=100, table_size=1e6, alpha=0.025, negative=5,
                 model_mode=1, train_mode=1, classes=10, itera=5):
        """
        :param train_file: 训练文件路径
        :param window: n-gram的窗口值
        :param min_reduce: 舍弃词频小于指定值的单词
        :param layer1_size: hs的大小
        :param table_size: neg的大小
        :param alpha: 学习率
        :param negative:
        :param model_mode: 1 -> CBOW, 0 -> skip_gram
        :param train_mode: 1 -> hs, 0 -> neg
        :param classes: 聚类的簇的数量
        :param itera: 迭代次数
        """
        self.train_file = train_file
        self.window = window
        self.min_reduce = min_reduce
        self.layer1_size = layer1_size
        self.table_size = int(table_size)
        self.alpha = alpha
        self.negative = negative
        self.model_mode = model_mode
        self.train_mode = train_mode
        self.classes = classes
        self.itera = itera
        self.ReadWord()
        self.SortVocab()
        self.ReduceVocab()
        if self.train_mode:
            self.CreateBinaryTree()
        else:
            self.InitUnigramTable()
        # 随机初始化词向量，矩阵大小 => [vocab_size, layer1_size]
        self.syn0 = np.random.random((self.vocab_size, self.layer1_size))
        # 初始化权重矩阵
        self.syn1 = np.zeros((self.vocab_size, self.layer1_size))
        # 词向量矩阵和
        self.neu1 = np.zeros((self.layer1_size))
        for i in range(self.itera):
            if self.model_mode:
                self.CBOW()
            else:
                self.skip_gram()
            print('第%d次迭代' % (i + 1))

    # 读单词
    def ReadWord(self):
        self.sen = []
        for line in open(self.train_file, 'r', encoding='utf-8'):
            self.sen.append(line.replace('\n', ''))

    # 按词频排序，制作字典
    def SortVocab(self):
        self.word_list = []
        for line in self.sen:
            for word in jieba.cut(line):
                if word.isalpha():
                    self.word_list.append(word)
        self.word_dict = collections.Counter(self.word_list)

    # 低频词的处理
    def ReduceVocab(self):
        word_dict = self.word_dict.copy()
        for k in self.word_dict.keys():
            if self.word_dict[k] < self.min_reduce:
                del word_dict[k]
        self.word_dict = word_dict

    # 建立哈夫曼树
    def CreateBinaryTree(self):
        # 字典从大到小排序，以便创建哈夫曼树
        self.word_dict = sorted(self.word_dict.items(), key=lambda x: x[1], reverse=True)
        self.word = []
        self.cn = []
        for (k, v) in self.word_dict:
            self.word.append(k)
            self.cn.append(v)
        self.vocab_size = len(self.word)
        # 哈夫曼树用数组形式表示，[:vocab_size]保存单词，[vocab_size:]保存父节点
        # 从vocab_size开始查找，因为已经按词频排序，所以从中间相两边比较即可创建哈夫曼树
        pos1 = self.vocab_size - 1
        pos2 = self.vocab_size
        # 源码矩阵大小为2*vocab_size+1，存疑
        count = [-1 for _ in range(2 * self.vocab_size - 1)]  # 保存结点的值
        count[:self.vocab_size] = self.cn
        count[self.vocab_size:] = [1e15 for _ in range(self.vocab_size - 1)]
        parent_node = ['' for _ in range(2 * self.vocab_size - 2)]  # 保存父节点的下标
        binary = [0 for _ in range(2 * self.vocab_size - 1)]  # 保存编码
        # 哈夫曼树的总结点数为2*vocab_size-1
        for a in range(self.vocab_size - 1):
            # 最小值和次最小值
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1
            # 从vocab_size开始依次保存父节点的值
            count[self.vocab_size + a] = count[min1] + count[min2]
            parent_node[min1] = self.vocab_size + a
            parent_node[min2] = self.vocab_size + a
            # 次最小值即右孩子赋1
            binary[min2] = 1
        # 哈夫曼编码
        self.codelen = ['' for _ in range(self.vocab_size)]  # 保存编码长度
        self.code = [[] for _ in range(self.vocab_size)]  # 保存哈夫曼编码
        self.point = [[] for _ in range(self.vocab_size)]  # 保存父节点
        for a in range(self.vocab_size):
            b = a
            i = 0
            while b < self.vocab_size * 2 - 2:
                self.code[a].append(binary[b])
                self.point[a].append(b - self.vocab_size)
                i += 1
                b = parent_node[b]
            self.codelen[a] = i
            self.code[a] = self.code[a][::-1]
            self.point[a] = self.point[a][::-1]
        print('CreateBinaryTree successful!')

    def InitUnigramTable(self):
        self.word = []
        self.cn = []
        for (k, v) in self.word_dict.items():
            self.word.append(k)
            self.cn.append(v)
        self.vocab_size = len(self.word)
        power = 0.75
        self.table = np.zeros(self.table_size, dtype=np.int64)
        train_words_pow = 0
        for a in range(self.vocab_size):
            train_words_pow += np.power(self.cn[a], power)
        i = 0  # 单词下标
        d1 = np.power(self.cn[i], power)
        for a in range(self.table_size):
            self.table[a] = i
            # 把table按词频划分，词频越高，占table的位置越多
            if a / self.table_size > d1:
                i += 1
                d1 += np.power(self.cn[i], power) / train_words_pow
            if i >= self.vocab_size:
                i = self.vocab_size - 1
        print('InitUnigramTable successful!')

    def CBOW(self):
        for i, word in enumerate(self.word):
            # print(i,word)
            neu1e = np.zeros((self.layer1_size))
            # in -> hidden
            # 对指定单词前后window个单词的权值进行更新，平均池化
            b = np.random.randint(self.window)
            for a in range(b, self.window * 2 + 1 + b):
                l1 = i - self.window + a
                if l1 < 0:
                    continue
                if l1 >= self.vocab_size:
                    continue
                for b in range(self.layer1_size):
                    self.neu1[b] += self.syn0[l1][b] / (self.window * 2 + 1)
            # 训练自身隐藏层结点权值、自身词向量更新系数
            if self.train_mode:
                for d in range(self.codelen[i]):
                    f = 0
                    # 路径上的点的序号
                    l2 = self.point[i][d]
                    # 小于0为叶结点，即单词自身，不迭代
                    if l2 < 0:
                        continue
                    # 计算f
                    for b in range(self.layer1_size):
                        f += self.neu1[b] * self.syn1[l2][b]
                    # sigmoid function
                    f = 1.0 / (1.0 + np.exp(-f))
                    # 计算学习率
                    g = (1 - self.code[i][d] - f) * self.alpha
                    # 记录累积误差项
                    for b in range(self.layer1_size):
                        neu1e[b] += g * self.syn1[l2][b]
                    # 更新非叶结点权重
                    for b in range(self.layer1_size):
                        self.syn1[l2][b] += g * self.neu1[b]
            else:
                # 随机采个数最多为negative的负样本
                for d in range(self.negative + 1):
                    # 第一个采样该单词，为正样本，其余采样为负样本
                    if d == 0:
                        l2 = i
                        label = 1
                    else:
                        rand = np.random.randint(self.table_size)
                        l2 = self.table[rand]
                        # 若采样落在该单词占有的区域，则跳过
                        # 词频越高，跳过几率越大，最终采到的样本越少
                        if l2 == i:
                            continue
                        label = 0
                    f = 0
                    # 计算f
                    for b in range(self.layer1_size):
                        f += self.neu1[b] * self.syn1[l2][b]
                    # sigmoid function
                    f = 1.0 / (1.0 + np.exp(-f))
                    # 计算学习率
                    g = (label - f) * self.alpha
                    # 记录累积误差项
                    for b in range(self.layer1_size):
                        neu1e[b] += g * self.syn1[l2][b]
                    # 更新非叶结点权重
                    for b in range(self.layer1_size):
                        self.syn1[l2][b] += g * self.neu1[b]
            # hidden -> in
            # 更新词向量，把选中的训练好的词向量系数加到其他词的向量上
            for a in range(self.window * 2 + 1):
                l1 = i - self.window + a
                if l1 < 0:
                    continue
                if l1 >= self.vocab_size:
                    continue
                for b in range(self.layer1_size):
                    self.syn0[l1][b] += neu1e[b]

    # TODO 耗时长
    def skip_gram(self):
        for i, word in enumerate(self.word):
            # print(i,word)
            neu1e = np.zeros((self.layer1_size))
            b = np.random.randint(self.window)
            for a in range(b, 2 * self.window + 1 + b):
                l1 = i - self.window + a
                if l1 < 0:
                    continue
                if l1 >= self.vocab_size:
                    continue
                if self.train_mode:
                    for d in range(self.codelen[i]):
                        f = 0
                        l2 = self.point[i][d]
                        if l2 < 0:
                            continue
                        # hidden -> out
                        for b in range(self.layer1_size):
                            f += self.syn0[l1][b] * self.syn1[l2][b]
                        # sigmoid function
                        f = 1.0 / (1.0 + np.exp(-f))
                        # 计算学习率
                        g = (1 - self.code[i][d] - f) * self.alpha
                        # 记录累积误差项
                        for b in range(self.layer1_size):
                            neu1e[b] += g * self.syn1[l2][b]
                        # 更新非叶结点权重
                        for b in range(self.layer1_size):
                            self.syn1[l2][b] += g * self.syn0[l1][b]
                else:
                    for d in range(self.negative + 1):
                        # 第一个采样该单词，为正样本，其余采样为负样本
                        if d == 0:
                            l2 = i
                            label = 1
                        else:
                            rand = np.random.randint(self.table_size)
                            l2 = self.table[rand]
                            # 若采样落在该单词占有的区域，则跳过
                            # 词频越高，跳过几率越大，最终采到的样本越少
                            if l2 == i:
                                continue
                            label = 0
                        f = 0
                        # hidden -> out
                        for b in range(self.layer1_size):
                            f += self.syn0[l1][b] * self.syn1[l2][b]
                        # sigmoid function
                        f = 1.0 / (1.0 + np.exp(-f))
                        # 计算下降梯度
                        g = (label - f) * self.alpha
                        # 记录累积误差项
                        for b in range(self.layer1_size):
                            neu1e[b] += g * self.syn1[l2][b]
                        # 更新非叶结点权重
                        for b in range(self.layer1_size):
                            self.syn1[l2][b] += g * self.syn0[l1][b]
                # in -> hidden
                # 把其他词训练好的误差向量加到选中词的向量上
                for b in range(self.layer1_size):
                    self.syn0[l1][b] += neu1e[b]
        del self.table

    def kmeans(self):
        pass

    def most_similar(self, s, cn=10):
        ind = self.word.index(s)
        sim = []
        for i in range(self.vocab_size):
            sim.append(np.dot(self.syn0[ind], self.syn0[i]))
        sim_ = sim.copy()
        sim_.sort()
        similar = []
        for i in range(cn):
            ind = sim.index(sim_[i])
            similar.append(self.word[ind])
        return similar

    # 求词的模
    def value(self, s):
        ind = self.word.index(s)
        return np.sqrt((self.syn0[ind] ** 2).sum())


if __name__ == '__main__':
    filename = './data/Q.txt'
    word2vec = Word2vec(filename, model_mode=1, train_mode=1, itera=5)
    # print(word2vec.value('计算机'))
    print(word2vec.most_similar('笔记本', 10))
    sen = [word2vec.word_list, []]
    model = gensim.models.Word2Vec(sen, min_count=1, size=200, hs=1, sg=0)
    print(model.most_similar('笔记本'))
