import numpy as np


class Metrics:
    def confusion_matrix(self, y_true, y_pred):
        r = np.unique(y_true)
        m = np.zeros((len(r), len(r)))
        for i in range(len(r)):
            for j in range(len(r)):
                m[i, j] = np.sum(np.logical_and(y_true == r[i], y_pred == r[j]))

        return m, r

    def tp(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true == pos_label, y_pred == pos_label))

    def fp(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true != pos_label, y_pred == pos_label))

    def fn(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true == pos_label, y_pred != pos_label))

    def tn(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true != pos_label, y_pred != pos_label))

    def tpr(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true == pos_label, y_pred == pos_label)) / np.sum(y_true == pos_label)

    def fpr(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true != pos_label, y_pred == pos_label)) / np.sum(y_true != pos_label)

    def fnr(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true == pos_label, y_pred != pos_label)) / np.sum(y_true == pos_label)

    def tnr(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true != pos_label, y_pred != pos_label)) / np.sum(y_true != pos_label)

    def ppv(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true == pos_label, y_pred == pos_label)) / np.sum(y_pred == pos_label)

    def fdr(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true != pos_label, y_pred == pos_label)) / np.sum(y_pred == pos_label)

    def For(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true == pos_label, y_pred != pos_label)) / np.sum(y_pred != pos_label)

    def npv(self, y_true, y_pred, pos_label=1):
        return np.sum(np.logical_and(y_true != pos_label, y_pred != pos_label)) / np.sum(y_pred != pos_label)

    def acc(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def f_measure(self, y_true, y_pred, pos_label=1, alpha=1):
        p = self.ppv(y_true, y_pred, pos_label)
        r = self.tpr(y_true, y_pred, pos_label)
        return (1 + alpha) * p * r / (alpha ** 2 * p + r)

    def roc(self, y_true, y_pred, pos_label=1):
        tpr = self.tpr(y_true, y_pred, pos_label)
        fpr = self.fpr(y_true, y_pred, pos_label)
        return tpr / fpr

    def roc_curve(self, y_true, y_score, pos_label=1):
        scores = np.unique(y_score)[::-1]
        tpr = np.zeros(len(scores) + 1)
        fpr = np.zeros(len(scores) + 1)
        threshold = np.zeros(len(scores) + 1)
        threshold[0] = scores[0] + 1
        for i, score in enumerate(scores):
            y_pred = np.zeros_like(y_score)
            y_pred[y_score >= score] = pos_label
            y_pred[y_score < score] = pos_label - 1
            tpr[i + 1] = self.tpr(y_true, y_pred, pos_label)
            fpr[i + 1] = self.fpr(y_true, y_pred, pos_label)
            threshold[i + 1] = score

        return fpr, tpr, threshold

    def auc(self, y_true, y_score, pos_label=1):
        pos = np.where(y_true == pos_label)[0]
        neg = np.where(y_true != pos_label)[0]
        s = 0
        for i in pos:
            for j in neg:
                if y_score[i] > y_score[j]:
                    s += 1
                elif y_score[i] == y_score[j]:
                    s += .5

        return s / (len(pos) * len(neg))

    def mae(self, y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / len(y_true)

    def mse(self, y_true, y_pred):
        return np.sum((y_true - y_pred) ** 2) / len(y_true)

    def r_square(self, y_true, y_pred):
        mean = np.mean(y_true)
        return 1 - (y_true - y_pred) ** 2 / (y_true - mean) ** 2

    def adjusted_r_square(self, y_true, y_pred, n):
        r = self.r_square(y_true, y_pred)
        N = len(y_true)
        return 1 - (1 - r) * (N - 1) / (N - n - 1)

    def hopkins_statistics(self, x, n=10):
        def get_r(a):
            """求每个样本到到样本空间中最近点的距离之和"""
            min_distance = np.inf
            dist = []
            for i in a:
                for j in x:
                    if np.array_equal(i, j):
                        continue

                    distance = np.linalg.norm(j - i)
                    min_distance = min(distance, min_distance)

                dist.append(min_distance)

            return np.sum(dist)

        a1 = x[np.random.randint(0, len(x), n)]
        a2 = x[np.random.randint(0, len(x), n)]

        r1 = get_r(a1)
        r2 = get_r(a2)

        return r2 / (r1 + r2)


metrics = Metrics()

if __name__ == '__main__':
    np.random.seed(6)
    y_true = np.array([1, 1, 2, 2])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    print(metrics.auc(y_true, y_pred, pos_label=2))

    from sklearn import metrics

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
    print(metrics.auc(fpr, tpr))
