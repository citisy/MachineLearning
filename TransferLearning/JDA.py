import numpy as np
import scipy
import sklearn
from sklearn.metrics import roc_curve


class JDA:
    def __init__(self, kernel_type='primal', gamma=1):
        """
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        """
        self.kernel_type = kernel_type,
        self.gamma = gamma
        self.classifier = None
        self.A = None

    def fit(self, x_source, x_target, y_source, y_target=None,
            dim=100, lamb=1, T=10, **kwargs):
        """
        Transform and Predict using 1NN as JDA paper did
        :param x_source: ns * n_feature, source feature
        :param y_source: ns * 1, source label
        :param x_target: nt * n_feature, target feature
        :param y_target: nt * 1, target label
        :return: acc, y_pred, list_acc
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param T: iteration number
        """
        X = np.hstack((x_source.T, x_target.T))
        # X /= np.linalg.norm(X, axis=0)    # 归一化
        m, n = X.shape
        ns, nt = len(x_source), len(x_target)
        e1 = 1 / ns * np.ones((ns, 1))
        e2 = 1 / nt * np.ones((nt, 1))
        e = np.vstack((e1, e2))
        C = len(np.unique(y_source))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        y_target_pseudo = None
        best_model = None
        best_metrics = -1
        cache_metrics = -1
        best_A = -1

        for t in range(T):
            M = np.dot(e, e.T) * C
            if y_target_pseudo is not None and len(y_target_pseudo) == nt:
                for c in range(C):
                    e1 = np.zeros((ns, 1))
                    e1[y_source == c] = 1 / len(y_source[y_source == c])
                    e2 = np.zeros((nt, 1))
                    e2[y_target_pseudo == c] = -1 / len(y_target_pseudo[y_target_pseudo == c])
                    e = np.vstack((e1, e2))
                    e[np.isinf(e)] = 0
                    M = M + np.dot(e, e.T)

            M = M / np.linalg.norm(M, 'fro')
            K = self.kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:dim]]
            Z = np.dot(K.T, A)
            # Z /= np.linalg.norm(Z, axis=0)    # 归一化
            x_source_new, x_target_new = Z[:ns], Z[ns:]

            clf = self.train_classify(x_source_new, y_source.ravel())

            y_target_pseudo = clf.predict(x_target_new)
            y_pred = clf.predict_proba(x_target_new)[:, 1]

            metrics = self.calculate_metrics(y_target, y_pred)

            print(f'JDA iteration [{t + 1}/{T}]: metrics: {metrics:.4f}')

            if best_metrics < metrics:
                best_metrics = metrics
                best_model = clf
                best_A = A

            if abs(cache_metrics - metrics) < 1e-4:
                break

            cache_metrics = metrics

        self.classifier, self.A = best_model, best_A

    def train_classify(self, trans_data, trans_label):
        """选择基分类器"""
        # 使用lightgbm作为基分类器
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(max_depth=3, num_leaves=16, class_weight='balanced')
        clf.fit(trans_data, trans_label)

        return clf

    def calculate_metrics(self, y_true, y_pred):
        """计算模型的评估指标"""
        # 使用ks作为评估指标
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return (tpr - fpr).max()

    def predict(self, x, **kwargs):
        x = self.kernel(self.kernel_type, x, None, gamma=self.gamma)
        x = np.dot(x, self.A)

        return self.classifier.predict(x, **kwargs)

    def predict_proba(self, x, **kwargs):
        x = self.kernel(self.kernel_type, x, None, gamma=self.gamma)
        x = np.dot(x, self.A)

        return self.classifier.predict_proba(x, **kwargs)

    def kernel(self, ker, X1, X2, gamma):
        K = None
        if not ker or ker == 'primal':
            K = X1
        elif ker == 'linear':
            if X2:
                K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
            else:
                K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
        elif ker == 'rbf':
            if X2:
                K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
            else:
                K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
        return K
