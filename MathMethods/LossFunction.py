import numpy as np


class Loss:
    def l1_loss(self, real, pred):
        return np.abs(pred - real)

    def d_l1_loss(self, real, pred):
        return np.where(pred < real, np.zeros_like(pred) - 1, np.ones_like(pred))

    def l2_loss(self, real, pred):
        return np.square(pred - real)

    def d_l2_loss(self, real, pred):
        return pred - real

    def huber_loss(self, real, pred, d=1):
        a = np.abs(real - pred)
        return np.where(a <= d, 0.5 * (real - pred) ** 2, d * a - 0.5 * d ** 2)

    def d_huber_loss(self, real, pred, d=1):
        r = np.zeros_like(pred) + d
        a = real - pred
        r[a > -d] = 0.5 * (pred - real)[a > -d]
        r[a > d] = -d
        return r

    def log_cosh_loss(self, real, pred):
        return np.log(np.cosh(pred - real))

    def d_log_cosh_loss(self, real, pred):
        return -np.tanh(pred - real)

    def quantile_loss(self, real, pred, g=.3):
        a = np.abs(real - pred)
        return np.where(real < pred, (1 - g) * a, g * a)

    def d_quantile_loss(self, real, pred, g=.3):
        return np.where(real < pred, 1 - g, g)

    def cross_entropy_loss(self, real, pred):
        return -real * np.log(pred)

    def d_cross_entropy_loss(self, real, pred):
        return -real / pred

    # def d_softmax_cross_entropy_loss(self, real, pred, axis=0):
    #     """equals (dcross_entropy_loss * dsoftmax)"""
    #     pred = np.array(pred)
    #     pred[np.where(real == np.max(real, axis=axis))] -= 1
    #     return pred

    def KL_loss(self, real, pred):
        return real * np.log(real / pred)

    def d_KL_loss(self, real, pred):
        return -real / pred

    def exp_loss(self, real, pred):
        return np.exp(-real * pred)

    def d_exp_loss(self, real, pred):
        return -real * np.exp(-real * pred)

    def Hinge_loss(self, real, pred):
        return np.maximum(0, 1 - real * pred)

    def d_Hinge_loss(self, real, pred):
        a = 1 - real * pred
        return np.where(a > 0, -real, np.zeros_like(a))

    def perceptron_loss(self, real, pred):
        return np.maximum(0, -pred)

    def d_perceptron_loss(self, real, pred):
        return np.where(pred > 0, np.zeros_like(pred), np.zeros_like(pred) - 1)


loss = Loss()
