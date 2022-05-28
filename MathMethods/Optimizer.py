import numpy as np


class Optimizer:
    def GradientDescent(self, vars, delta, lr=0.01):
        return vars - lr * delta

    def Momentum(self, vars, delta, mv=None, lr=0.01, discount=0.9):
        if mv is None:
            mv = np.zeros_like(delta)
        mv = mv * discount + lr * delta
        return vars - mv, mv

    def AdaGrad(self, vars, delta, r=None, eps=1e-6, lr=0.1):
        if r is None:
            r = np.zeros_like(delta)
        r += np.square(delta)
        return vars - delta * lr / np.sqrt(r + eps), r

    def Adadelta(self, vars, delta, r=None, v=None, lr=1, rho=0.95, eps=1e-6):
        if r is None:
            r = np.zeros_like(delta)
        if v is None:
            v = np.zeros_like(delta)
        v = rho * v + (1 - rho) * np.square(delta)
        delta = np.sqrt(r + eps) / np.sqrt(v + eps) * delta
        return vars - lr * delta, (rho * r + (1 - rho) * np.square(delta), v)

    def RMSProp(self, vars, delta, r=None, rho=0.9, lr=0.01, eps=1e-6):
        if r is None:
            r = np.zeros_like(delta)
        r = rho * r + (1 - rho) * np.square(delta)
        return vars - lr / np.sqrt(r + eps) * delta, r

    def Adam(self, vars, delta, m=None, v=None, t=1, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-6):
        """momentum + RMSProp"""
        if m is None:
            m = np.zeros_like(delta)
        if v is None:
            v = np.zeros_like(delta)
        m = beta1 * m + (1 - beta1) * delta  # momentum
        v = beta2 * v + (1 - beta2) * np.square(delta)  # RMSProp
        m_ = m / (1 - np.power(beta1, t))
        v_ = v / (1 - np.power(beta2, t))
        return vars - lr * m_ / (np.sqrt(v_) + eps), (m, v, t + 1)

    def AdaMax(self, vars, delta, m=None, v=None, t=1, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-6):
        if m is None:
            m = np.zeros_like(delta)
        if v is None:
            v = np.zeros_like(delta)
        m = beta1 * m + (1 - beta1) * delta  # momentum
        v = np.maximum(beta2 * v, np.abs(delta))
        m_ = m / (1 - np.power(beta1, t))
        return vars - lr * m_ / (np.sqrt(v) + eps), (m, v, t + 1)


optimizer = Optimizer()
