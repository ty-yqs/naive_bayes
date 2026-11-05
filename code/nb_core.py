"""
nb_core.py
==========
从零实现的（离散）朴素贝叶斯分类器，以及教材/论文中推导所对应的
全部统计量（先验、条件概率表、对数后验）。

记号与报告第 1 节保持一致：
    * 特征 X_1,...,X_m，第 i 个特征取值个数为 k_i，取值为 x_{i,1},...,x_{i,k_i}
    * 类别 Y 取值个数为 k，取值为 y_1,...,y_k
    * 训练集 D = {(x^(1), y^(1)), ..., (x^(N), y^(N))}

模型：
    pi_c      = P(Y = y_c)              （先验，用频数 + 平滑估计）
    theta_cji = P(X_i = x_{i,j} | Y = y_c)  （条件概率表）
    y_hat(x)  = argmax_c [ log pi_c + sum_i log theta_{c,i,x_i} ]

拉普拉斯平滑（加 alpha 平滑，alpha = 1 时即经典的“加一平滑”）：
    theta_cji = (N_cji + alpha) / (N_c + alpha * k_i)
其中 N_cji 是训练集中“类别为 c 且第 i 个特征取第 j 个值”的样本数，
N_c = sum_j N_cji 是类别 c 的样本数。
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def _as_arrays(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X 必须是二维数组 (N, m)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X 与 y 的样本数不一致")
    return X, y


class CategoricalNB:
    """多类别、多值离散特征的朴素贝叶斯分类器（对数空间实现）。"""

    def __init__(self, alpha: float = 1.0, class_prior_smoothing: bool = True):
        """
        参数
        ----
        alpha : 条件概率的拉普拉斯平滑系数（>= 0）。alpha=0 表示不做平滑（极大似然）。
        class_prior_smoothing : 先验是否也做加一平滑。样本量足够时几乎无影响。
        """
        if alpha < 0:
            raise ValueError("alpha 必须非负")
        self.alpha = float(alpha)
        self.class_prior_smoothing = class_prior_smoothing

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X, y = _as_arrays(X, y)
        n, m = X.shape

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        self.n_features_ = m

        # 每个特征的实际取值集合（可能不是 0..k_i-1 的连续整数）
        self.categories_ = [np.unique(X[:, i]) for i in range(m)]
        # 每个特征取值->列下标的映射，便于向量化查表
        self._cat_index = [
            {v: j for j, v in enumerate(cats)} for cats in self.categories_
        ]

        # ---------- 1. 类别先验 pi_c ----------
        N_c = np.bincount(y_idx, minlength=k).astype(float)
        if self.class_prior_smoothing:
            # 加一平滑，保证 log 有定义
            self.class_log_prior_ = np.log((N_c + 1.0) / (N_c.sum() + k))
        else:
            self.class_log_prior_ = np.log(N_c / N_c.sum())
        self.class_count_ = N_c

        # ---------- 2. 条件概率表 theta_cji ----------
        # 用“类内计数”的列表保存，每类一个 (m, k_i) 的矩阵
        self.feature_count_ = []
        self.feature_log_prob_ = []
        for i in range(m):
            k_i = len(self.categories_[i])
            idx = np.array([self._cat_index[i][v] for v in X[:, i]])
            # counts[c, j] = N_cji
            counts = np.zeros((k, k_i))
            np.add.at(counts, (y_idx, idx), 1.0)
            self.feature_count_.append(counts)

            smoothed = counts + self.alpha
            denom = counts.sum(axis=1, keepdims=True) + self.alpha * k_i
            with np.errstate(divide="ignore"):
                log_prob = np.log(smoothed / denom)
            # alpha=0 且某格计数为 0 时 log 为 -inf；用极小值代替，保持数值稳定
            log_prob = np.where(np.isfinite(log_prob), log_prob, -1e10)
            self.feature_log_prob_.append(log_prob)

        # 预先把 log 先验和条件表拼成“查表矩阵”，供预测时一次性 gather
        return self

    # -------------------------------------------------------------- predict
    def _joint_log_likelihood(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        k = len(self.classes_)
        jll = np.tile(self.class_log_prior_, (n, 1))  # (n, k)

        for i in range(self.n_features_):
            # 把第 i 个特征的取值映射到列下标；未见过的取值跳过（等价于概率为 0 的退化情况）
            col = np.full(n, -1, dtype=int)
            key = self._cat_index[i]
            for r in range(n):
                col[r] = key.get(X[r, i], -1)
            logp = self.feature_log_prob_[i]  # (k, k_i)
            for c in range(k):
                mask = col >= 0
                jll[mask, c] += logp[c, col[mask]]
        return jll

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        jll = jll - jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)


class GaussianNB:
    """连续特征版本：P(X_i|Y=c) 用正态分布 N(mu_ci, sigma_ci^2) 建模。

    数值细节：16 个特征的量纲差异极大（Area ~ 1e5，ShapeFactor2 ~ 1e-3），
    直接在原始尺度上计算 log 密度会出现严重的大数相消/下溢。由于正态模型对
    特征的仿射变换是等价的（predictions 不变），这里先在内部做标准化再建模，
    既保持数学上完全等价，又保证数值稳定。
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = float(var_smoothing)

    def fit(self, X, y):
        X, y = _as_arrays(X, y)
        self._mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        self._sd = sd
        Z = (X - self._mu) / self._sd

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        self.theta_ = np.zeros((k, Z.shape[1]))
        self.var_ = np.zeros((k, Z.shape[1]))
        self.class_prior_ = np.zeros(k)
        self.class_count_ = np.bincount(y_idx, minlength=k).astype(float)

        eps = self.var_smoothing * Z.var(axis=0).max() if Z.size else 0.0
        for c in range(k):
            Zc = Z[y_idx == c]
            self.class_prior_[c] = len(Zc) / len(Z)
            self.theta_[c] = Zc.mean(axis=0)
            self.var_[c] = Zc.var(axis=0) + eps
        self.class_log_prior_ = np.log(self.class_prior_)
        self._log_norm = -0.5 * np.log(2.0 * np.pi * self.var_)
        return self

    def _joint_log_likelihood(self, X):
        Z = (np.asarray(X, dtype=float) - self._mu) / self._sd
        jll = np.zeros((Z.shape[0], len(self.classes_)))
        for c in range(len(self.classes_)):
            d = Z - self.theta_[c]
            jll[:, c] = (
                self.class_log_prior_[c]
                + self._log_norm[c].sum()
                - 0.5 * ((d * d) / self.var_[c]).sum(axis=1)
            )
        return jll

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# 评价指标
# ----------------------------------------------------------------------------
def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    pos = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[pos[t], pos[p]] += 1
    return cm


def classification_report(y_true, y_pred, labels=None) -> dict:
    """返回每类的 precision / recall / f1 以及宏平均、加权平均。"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels)
    per_class = {}
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[str(lab)] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }
    supports = np.array([per_class[str(l)]["support"] for l in labels], dtype=float)
    macro = {
        m: float(np.mean([per_class[str(l)][m] for l in labels]))
        for m in ("precision", "recall", "f1")
    }
    weighted = {
        m: float(
            np.sum([per_class[str(l)][m] * per_class[str(l)]["support"] for l in labels])
            / supports.sum()
        )
        for m in ("precision", "recall", "f1")
    }
    return {
        "labels": [str(l) for l in labels],
        "per_class": per_class,
        "macro": macro,
        "weighted": weighted,
        "accuracy": accuracy(y_true, y_pred),
        "support": int(len(y_true)),
    }
