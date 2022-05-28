import numpy as np


class Loss:
    def l1_loss(self, y_true, y_pred):
        return np.abs(y_pred - y_true)

    def d_l1_loss(self, y_true, y_pred):
        return np.where(y_pred < y_true, np.zeros_like(y_pred) - 1, np.ones_like(y_pred))

    def l2_loss(self, y_true, y_pred):
        return np.square(y_pred - y_true)

    def d_l2_loss(self, y_true, y_pred):
        return y_pred - y_true

    def huber_loss(self, y_true, y_pred, d=1):
        a = np.abs(y_true - y_pred)
        return np.where(a <= d, 0.5 * (y_true - y_pred) ** 2, d * a - 0.5 * d ** 2)

    def d_huber_loss(self, y_true, y_pred, d=1):
        r = np.zeros_like(y_pred) + d
        a = y_true - y_pred
        r[a > -d] = 0.5 * (y_pred - y_true)[a > -d]
        r[a > d] = -d
        return r

    def log_cosh_loss(self, y_true, y_pred):
        return np.log(np.cosh(y_pred - y_true))

    def d_log_cosh_loss(self, y_true, y_pred):
        return -np.tanh(y_pred - y_true)

    def quantile_loss(self, y_true, y_pred, g=.3):
        a = np.abs(y_true - y_pred)
        return np.where(y_true < y_pred, (1 - g) * a, g * a)

    def d_quantile_loss(self, y_true, y_pred, g=.3):
        return np.where(y_true < y_pred, 1 - g, g)

    def cross_entropy_loss(self, y_true, y_pred):
        return -y_true * np.log(y_pred)

    def d_cross_entropy_loss(self, y_true, y_pred):
        return -y_true / y_pred

    # def d_softmax_cross_entropy_loss(self, y_true, y_pred, axis=0):
    #     """equals (dcross_entropy_loss * dsoftmax)"""
    #     y_pred = np.array(y_pred)
    #     y_pred[np.where(y_true == np.max(y_true, axis=axis))] -= 1
    #     return y_pred

    def KL_loss(self, y_true, y_pred):
        return y_true * np.log(y_true / y_pred)

    def d_KL_loss(self, y_true, y_pred):
        return -y_true / y_pred

    def exp_loss(self, y_true, y_pred):
        return np.exp(-y_true * y_pred)

    def d_exp_loss(self, y_true, y_pred):
        return -y_true * np.exp(-y_true * y_pred)

    def Hinge_loss(self, y_true, y_pred):
        return np.maximum(0, 1 - y_true * y_pred)

    def d_Hinge_loss(self, y_true, y_pred):
        a = 1 - y_true * y_pred
        return np.where(a > 0, -y_true, np.zeros_like(a))

    def perceptron_loss(self, y_true, y_pred):
        return np.maximum(0, -y_pred)

    def d_perceptron_loss(self, y_true, y_pred):
        return np.where(y_pred > 0, np.zeros_like(y_pred), np.zeros_like(y_pred) - 1)


loss = Loss()
