import numpy as np


class Activation:
    def no_activation(self, x):
        return x

    def d_no_activation(self, x):
        return np.ones_like(x)

    def tanh(self, x):
        """[-2, 2] is linear interval"""
        # return np.tanh(x)
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def d_tanh(self, x):
        return 1 - self.tanh(x) ** 2

    def sigmoid(self, x):
        """[-6, 6] is linear interval"""
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        y = self.sigmoid(x)
        return y - y ** 2

    def fast_sigmoid(self, x, mins=-6, maxs=6, steps=.1):
        """cache (maxs - mins) / steps numbers
        it will be slow while it's first to use because of initialing"""
        try:
            self.sigmoid_table = self.sigmoid_table
        except AttributeError:
            self.sigmoid_table = self.sigmoid(np.arange(mins, maxs, steps))

        y = np.zeros_like(x)
        idx = np.logical_and(x >= mins, x < maxs)
        y[idx] = self.sigmoid_table[np.int_((x[idx] - mins) / steps)]
        y[x < mins] = 0
        y[x > maxs] = 1
        return y

    def d_fast_sigmoid(self, x, mins=-6, maxs=6, steps=.1):
        y = self.fast_sigmoid(x, mins, maxs, steps)
        return y - y ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))

    def leaky_relu(self, x):
        return np.maximum(0.1 * x, x)

    def d_leaky_relu(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x) + .1)

    def elu(self, x, a=.2):
        return np.where(x > 0, x, a * (np.exp(x) - 1))

    def d_elu(self, x, a=.2):
        y = self.elu(x, a)
        return np.where(x > 0, np.ones_like(x), y + a)

    def selu(self, x, a=1.67, l=1.05):
        return l * self.elu(x, a)

    def d_selu(self, x, a=1.67, l=1.05):
        return l * self.d_elu(x, a)

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        # return x * self.sigmoid(1.702 * x)

    def d_gelu(self, x):
        return (0.5 * np.tanh(0.0356774 * x ** 3 + 0.797885 * x)
                + (0.0535161 * x ** 3 + 0.398942 * x) * (1 / np.cosh(0.0356774 * x ** 3 + 0.797885 * x) ** 2)
                + 0.5)

    def swish(self, x):
        return x * self.sigmoid(x)

    def d_swish(self, x):
        y = self.swish(x)
        return y + self.sigmoid(x) * (1 - y)

    def softmax(self, x, axis=0):
        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

    def d_softmax(self, x, axis=0):
        y = self.softmax(x, axis)
        a = np.max(y, axis=axis)
        ia, ib = y == a, y != a
        y[ia] = y[ia] * (1 - y[ia])
        y[ib] = -y[ia] * y[ib]
        return y

    def dropout(self, x, drop_pro=.7):
        r = np.random.rand(*x.shape)
        return np.where(r > drop_pro, np.zeros_like(x), x)


activation = Activation()
