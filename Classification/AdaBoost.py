import numpy as np


def adaboost(data, label, w=None, itera=10):
    # todo: only 1-d data, how about 2-d data and even higher dim data???
    data = np.array(data, dtype=float)
    label = np.array(label, dtype=float)
    n_samples = data.shape[1]
    cache = []

    if w is None:
        w = np.zeros_like(label) + 1. / n_samples

    for i in range(itera):
        ems = []
        for j in np.arange(np.min(data[0]), np.max(data[0])):
            pre = np.ones_like(label)
            v = j + 0.5
            pre[np.where(data[0] >= v)] = -1
            e = np.sum(w[np.where(pre != label)])
            left = 1
            if e >= 0.5:
                left = -1
                e = 1 - e
            ems.append([v, e, left])

        ems = np.array(ems)
        argmin = np.argmin(ems[:, 1])
        alpha = 0.5 * np.log((1 - ems[argmin, 1]) / ems[argmin, 1])
        g = np.ones_like(label)
        if ems[argmin, 2] == 1:
            g[np.where(data[0] >= ems[argmin, 0])] = -1
        else:
            g[np.where(data[0] < ems[argmin, 0])] = -1

        w *= np.exp(-alpha * label * g)
        w /= np.sum(w)
        cache.append([alpha, g])

    fx = 0
    for i in cache:
        fx += i[0] * i[1]

    return np.sign(fx)


def make_data():
    x = [list(range(10))]
    y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    return x, y


if __name__ == '__main__':
    x, y = make_data()
    pre = adaboost(x, y, itera=3)
    print(pre)
    print('acc: %.2f' % (np.sum(pre == y) * 1. / len(y)))
