"""
run_step3.py
============
第 3 步：思考问题。

(1) 连续特征版本的朴素贝叶斯：高斯朴素贝叶斯 + 核密度（KDE）朴素贝叶斯，
    与第 2 步的离散化版本比较。
(2) 独立性假设的合理性：特征相关性分析 + “重复使用相关特征”实验
    （把某个特征复制若干份，等价于人为提高它的权重）。

使用原始连续特征（bean_raw.csv，partition == TRAIN / TEST）。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_core import (  # noqa: E402
    CategoricalNB,
    GaussianNB,
    accuracy,
    classification_report,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# KDE 朴素贝叶斯：P(X_i = x | Y = c) 用一维核密度估计
# ---------------------------------------------------------------------------
class KernelNB:
    """一维高斯核密度估计的朴素贝叶斯（对数密度，网格 + 线性插值加速）。"""

    def __init__(self, bandwidth="scott", grid_size=1024):
        self.bandwidth = bandwidth
        self.grid_size = grid_size

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k, m = len(self.classes_), X.shape[1]
        self.class_log_prior_ = np.zeros(k)
        self._grids = [[None] * m for _ in range(k)]

        for c in range(k):
            Xc = X[y_idx == c]
            self.class_log_prior_[c] = np.log(len(Xc) / len(X))
            for i in range(m):
                v = np.sort(Xc[:, i])
                n = len(v)
                std = v.std(ddof=1) if n > 1 else 1.0
                iqr = np.subtract(*np.percentile(v, [75, 25]))
                sigma = min(std, iqr / 1.349) if iqr > 0 else std
                h = 1.06 * sigma * n ** (-1 / 5)  # Scott/Silverman 经验带宽
                if not np.isfinite(h) or h <= 0:
                    h = max(std, 1e-6)
                lo, hi = v[0] - 3 * h, v[-1] + 3 * h
                grid = np.linspace(lo, hi, self.grid_size)
                # log density = logsumexp(-0.5 z^2) - log(n) - log(h*sqrt(2pi))
                z = (grid[:, None] - v[None, :]) / h
                with np.errstate(divide="ignore"):
                    ld = _logsumexp(-0.5 * z * z, axis=1)
                ld = ld - np.log(n) - np.log(h * np.sqrt(2 * np.pi))
                self._grids[c][i] = (grid, ld)
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        k = len(self.classes_)
        jll = np.tile(self.class_log_prior_, (n, 1))
        for c in range(k):
            for i in range(X.shape[1]):
                grid, ld = self._grids[c][i]
                jll[:, c] += np.interp(X[:, i], grid, ld, left=ld[0], right=ld[-1])
        return jll

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


def _logsumexp(a, axis=1):
    m = a.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def main():
    res = {}
    raw = pd.read_csv(os.path.join(ROOT, "bean_raw.csv"))
    feat_names = [c for c in raw.columns if c not in ("id", "partition",
                                                      "in_train_1000", "Class")]
    tr = raw[raw.partition == "TRAIN"]
    te = raw[raw.partition == "TEST"]
    Xtr, ytr = tr[feat_names].to_numpy(float), tr["Class"].to_numpy()
    Xte, yte = te[feat_names].to_numpy(float), te["Class"].to_numpy()
    res["data"] = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
                   "features": feat_names}

    # ---------- (1a) 高斯朴素贝叶斯（原始连续特征） ----------
    gnb = GaussianNB().fit(Xtr, ytr)
    p_gnb = gnb.predict(Xte)
    rep_gnb = classification_report(yte, p_gnb)
    res["gaussian_nb"] = {"accuracy": rep_gnb["accuracy"], "report": rep_gnb}

    # ---------- (1b) KDE 朴素贝叶斯 ----------
    knb = KernelNB().fit(Xtr, ytr)
    p_knb = knb.predict(Xte)
    rep_knb = classification_report(yte, p_knb)
    res["kernel_nb"] = {"accuracy": rep_knb["accuracy"], "report": rep_knb}

    # ---------- (1c) 对数变换后再用高斯 NB ----------
    Xtr_log = np.log1p(np.abs(Xtr)) * np.sign(Xtr)
    Xte_log = np.log1p(np.abs(Xte)) * np.sign(Xte)
    gnb_log = GaussianNB().fit(Xtr_log, ytr)
    res["gaussian_nb_log"] = {
        "accuracy": accuracy(yte, gnb_log.predict(Xte_log))
    }

    # ---------- (1d) 特征标准化后的高斯 NB（与 1a 等价，验证尺度不变性） ----------
    mu, sd = Xtr.mean(0), Xtr.std(0)
    gnb_s = GaussianNB().fit((Xtr - mu) / sd, ytr)
    res["gaussian_nb_standardized"] = {
        "accuracy": accuracy(yte, gnb_s.predict((Xte - mu) / sd))
    }

    # ---------- (2a) 特征相关性 ----------
    corr = np.corrcoef(Xtr, rowvar=False)
    res["correlation"] = {"features": feat_names, "matrix": corr.tolist()}
    pairs = []
    for i in range(len(feat_names)):
        for j in range(i + 1, len(feat_names)):
            pairs.append({"a": feat_names[i], "b": feat_names[j],
                          "r": float(corr[i, j])})
    pairs.sort(key=lambda d: -abs(d["r"]))
    res["top_correlated_pairs"] = pairs[:12]
    res["n_pairs_abs_r_gt_09"] = sum(1 for p in pairs if abs(p["r"]) > 0.9)
    res["n_pairs_abs_r_gt_099"] = sum(1 for p in pairs if abs(p["r"]) > 0.99)

    # ---------- (2b) 离散特征上的“重复特征”实验 ----------
    trd = pd.read_csv(os.path.join(ROOT, "bean_train.csv"))
    ted = pd.read_csv(os.path.join(ROOT, "bean_test.csv"))
    fnames_d = [c for c in trd.columns if c not in ("id", "Class")]
    Xd_tr = trd[fnames_d].to_numpy()
    Xd_te = ted[fnames_d].to_numpy()
    yd_tr, yd_te = trd["Class"].to_numpy(), ted["Class"].to_numpy()

    def dup_experiment(cols, reps, alpha=1.0):
        idx = [fnames_d.index(c) for c in cols] if cols else []
        Xa = [Xd_tr]
        Xb = [Xd_te]
        for i in idx:
            for _ in range(reps):
                Xa.append(Xd_tr[:, [i]])
                Xb.append(Xd_te[:, [i]])
        A, B = np.hstack(Xa), np.hstack(Xb)
        mdl = CategoricalNB(alpha=alpha).fit(A, yd_tr)
        pred = mdl.predict(B)
        return accuracy(yd_te, pred), mdl, B

    dup = []
    for reps in [1, 2, 3, 4, 6, 8, 12]:
        acc, mdl, B = dup_experiment(["Area"], reps - 1)
        # 平均最大后验概率（置信度）
        conf = float(mdl.predict_proba(B).max(axis=1).mean())
        dup.append({"reps": reps, "accuracy": acc, "mean_max_prob": conf})
    res["duplicate_area"] = dup

    dup2 = []
    for reps in [1, 2, 3, 4, 6]:
        acc, mdl, B = dup_experiment(["Area", "EquivDiameter"], reps - 1)
        conf = float(mdl.predict_proba(B).max(axis=1).mean())
        dup2.append({"reps": reps, "accuracy": acc, "mean_max_prob": conf})
    res["duplicate_area_equivdiameter"] = dup2

    # ---------- (2c) 去掉冗余特征 / 只保留一组 ----------
    groups = {
        "drop_Area": "Area",                       # 保留 EquivDiameter
        "drop_EquivDiameter": "EquivDiameter",     # 保留 Area
        "drop_ConvexArea": "ConvexArea",
        "drop_Perimeter": "Perimeter",
        "drop_MajorAxisLength": "MajorAxisLength",
        "drop_Eccentricity": "Eccentricity",       # 保留 AspectRation
        "drop_Compactness": "Compactness",
        "drop_Eccentricity_and_Compactness": None,
    }
    drop_res = {}
    for name, col in groups.items():
        keep = [i for i, c in enumerate(fnames_d)
                if c != col and not (name.endswith("and_Compactness")
                                     and c == "Compactness")]
        mdl = CategoricalNB(alpha=1.0).fit(Xd_tr[:, keep], yd_tr)
        drop_res[name] = {"n_features": len(keep),
                          "accuracy": accuracy(yd_te, mdl.predict(Xd_te[:, keep]))}
    res["drop_features"] = drop_res

    # ---------- (2d) 高斯 NB 上再叠加完全相关的特征 ----------
    # 用 Area 与其线性变换（等价于完全相关的新特征）测试“重复计数”的影响
    X2_tr = np.hstack([Xtr, Xtr[:, [0]] * 2.0])
    X2_te = np.hstack([Xte, Xte[:, [0]] * 2.0])
    g2 = GaussianNB().fit(X2_tr, ytr)
    res["gaussian_nb_dup_area"] = {
        "accuracy": accuracy(yte, g2.predict(X2_te)),
        "mean_max_prob": float(g2.predict_proba(X2_te).max(axis=1).mean()),
    }
    res["gaussian_nb_mean_max_prob"] = float(
        gnb.predict_proba(Xte).max(axis=1).mean()
    )

    # ---------- (2e) 逐特征删除对离散 NB 的影响 ----------
    loo = {}
    for i, name in enumerate(fnames_d):
        keep = [j for j in range(len(fnames_d)) if j != i]
        mdl = CategoricalNB(alpha=1.0).fit(Xd_tr[:, keep], yd_tr)
        loo[name] = accuracy(yd_te, mdl.predict(Xd_te[:, keep]))
    res["leave_one_feature_out"] = loo

    with open(os.path.join(OUT, "step3.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("Gaussian NB      :", round(res["gaussian_nb"]["accuracy"], 4))
    print("KDE NB           :", round(res["kernel_nb"]["accuracy"], 4))
    print("Gaussian NB log  :", round(res["gaussian_nb_log"]["accuracy"], 4))
    print("Gaussian NB std  :", round(res["gaussian_nb_standardized"]["accuracy"], 4))
    print("dup Area         :", [(d["reps"], round(d["accuracy"], 4),
                                 round(d["mean_max_prob"], 3)) for d in dup])
    print("dup Area+EqD     :", [(d["reps"], round(d["accuracy"], 4),
                                 round(d["mean_max_prob"], 3)) for d in dup2])
    print("drop features    :", {k: round(v["accuracy"], 4) for k, v in drop_res.items()})
    print("gauss dup        :", res["gaussian_nb_dup_area"])
    print("top pairs        :", [(p["a"], p["b"], round(p["r"], 4))
                                 for p in res["top_correlated_pairs"][:8]])
    print("|r|>0.9          :", res["n_pairs_abs_r_gt_09"], " |r|>0.99:",
          res["n_pairs_abs_r_gt_099"])
    print("LOO worst/best   :", sorted(loo.items(), key=lambda kv: kv[1])[:3],
          sorted(loo.items(), key=lambda kv: -kv[1])[:3])


if __name__ == "__main__":
    main()
