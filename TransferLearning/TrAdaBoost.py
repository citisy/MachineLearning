import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc


class TrAdaBoost:
    def __init__(self):
        self.classifier = None

    def fit(self,
            x_source, y_source,
            x_target=None, y_targe=None,
            N=500, early_stopping_rounds=20, **kwargs):
        # ======= 初始化 =======
        # 拼接数据集
        trans_data = np.concatenate((x_source, x_target), axis=0)
        trans_label = np.concatenate((y_source, y_targe), axis=0)
        trans_len = x_source.shape[0] + x_target.shape[0]

        # 数据集样本数
        row_source = x_source.shape[0]
        row_target = x_target.shape[0]

        # 初始化权重
        weights_source, weights_target = self.init_weight(row_source, row_target)
        weights = np.concatenate((weights_source, weights_target), axis=0)

        # 按照公式初始化beta值
        beta = 1 / (1 + np.sqrt(2 * np.log(row_source / N)))

        # 存每一次迭代的beta值=error_rate / (1 - error_rate)
        beta_T = np.zeros([1, N])

        # 存储每次迭代的标签
        y_target_preds = np.ones([row_source + row_target, N])

        trans_data = np.asarray(trans_data, order='C')
        trans_label = np.asarray(trans_label, order='C')

        # 最优的评估指标
        best_metrics = -1
        # 最优基模型数量
        best_round = -1
        # 最优模型
        best_model = None

        # ======= 正式训练 =========
        for i in range(N):
            # 计算weight
            total = np.sum(weights)
            P = np.asarray(weights / total, order='C')

            y_target_preds[:, i], model = self.train_classify(trans_data, trans_label, P)
            y_target_pred = y_target_preds[row_source:row_source + row_target, i]
            pctg = np.sum(trans_label) / len(trans_label)
            thred = pd.DataFrame(y_target_pred).quantile(1 - pctg)[0]

            y_target_pred = np.where(y_target_pred <= thred, 0, 1)

            # 计算在目标域上的错误率
            error_rate = self.calculate_error_rate(y_targe, y_target_pred, weights[row_source:row_source + row_target, :])

            # 防止过拟合
            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                break

            beta_T[0, i] = error_rate / (1 - error_rate)

            # 调整目标域样本权重
            for j in range(row_target):
                weights[row_source + j] = weights[row_source + j] * np.power(beta_T[0, i], (-np.abs(y_target_preds[row_source + j, i] - y_targe[j])))

            # 调整源域样本权重
            for j in range(row_source):
                weights[j] = weights[j] * np.power(beta, np.abs(y_target_preds[j, i] - y_source[j]))

            metrics = self.calculate_metrics(y_targe, y_target_preds[(row_source + row_target):, i])

            print('metrics : ', metrics, 'error_rate : ', error_rate, '当前第', i + 1, '轮')

            # 不再使用后一半学习器投票，而是只保留效果最好的逻辑回归模型
            if metrics > best_metrics:
                best_metrics = metrics
                best_round = i
                best_model = model
            # 当超过early_stopping_rounds轮KS不再提升后，停止训练
            if best_round < i - early_stopping_rounds:
                break

        self.classifier = best_model

    def init_weight(self, row_source, row_target):
        """初始化权重"""
        # 源域权重=1/n，目标域权重=1/(m+n)
        weights_source = np.ones([row_source, 1]) / row_source
        weights_target = np.ones([row_target, 1]) / (row_source + row_target)

        return weights_source, weights_target


    def train_classify(self, trans_data, trans_label, P):
        """选择基分类器"""
        # 使用lightgbm作为基分类器
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(max_depth=3, num_leaves=16, class_weight='balanced')
        clf.fit(trans_data, trans_label, sample_weight=P[:, 0])

        return clf.predict_proba(trans_data)[:, 1], clf

    def calculate_error_rate(self, y_target_true, y_target_pred, weight):
        """计算在目标域上面的错误率"""
        total = np.sum(weight)
        return np.sum(weight[:, 0] / total * np.abs(y_target_true - y_target_pred))

    def calculate_metrics(self, y_true, y_pred):
        """计算模型的评估指标"""
        # 使用ks作为评估指标
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return (tpr - fpr).max()


