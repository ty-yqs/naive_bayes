"""
run_extra.py
============
补充分析（报告中引用的小实验）：

1. 训练集中“空单元格”的数量 —— 解释为什么需要拉普拉斯平滑；
2. 平滑系数 alpha 的作用随训练样本量 N 的变化；
3. 两个模型在测试集上预测不一致的样本数；
4. 朴素贝叶斯的置信度与准确率的差距（过自信）。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_core import CategoricalNB, accuracy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(7)


def load(name):
    df = pd.read_csv(os.path.join(ROOT, name))
    return df, df.drop(columns=["id", "Class"]).to_numpy(), df["Class"].to_numpy()


def main():
    res = {}
    tr_df, Xtr, ytr = load("bean_train.csv")
    tr1_df, Xtr1, ytr1 = load("bean_train_1000.csv")
    te_df, Xte, yte = load("bean_test.csv")

    # ---------- 1. 空单元格统计 ----------
    def zero_stats(X, y):
        m = CategoricalNB(alpha=1.0).fit(X, y)
        total, zero = 0, 0
        per_class = {}
        for c, name in zip(m.classes_, m.class_count_):
            pass
        for ci, cname in enumerate(m.classes_):
            z, t = 0, 0
            for i in range(X.shape[1]):
                cnt = m.feature_count_[i][ci]
                z += int((cnt == 0).sum())
                t += len(cnt)
            per_class[str(cname)] = {"zero_cells": z, "cells": t,
                                     "n_samples": int(m.class_count_[ci])}
            zero += z
            total += t
        return {"total_cells": total, "zero_cells": zero,
                "zero_fraction": zero / total, "per_class": per_class}

    res["zero_cells_full"] = zero_stats(Xtr, ytr)
    res["zero_cells_1000"] = zero_stats(Xtr1, ytr1)

    # ---------- 2. alpha 的作用随 N 变化 ----------
    grid = []
    for n in [20, 30, 50, 100, 200, 500, 1000, 2000, 9526]:
        row = {"n": n}
        for alpha in (0.0, 1.0):
            accs = []
            reps = 40 if n <= 2000 else 5
            for _ in range(reps):
                sub = (np.arange(len(ytr)) if n >= len(ytr)
                       else RNG.choice(len(ytr), n, replace=False))
                mdl = CategoricalNB(alpha=alpha).fit(Xtr[sub], ytr[sub])
                accs.append(accuracy(yte, mdl.predict(Xte)))
            row[f"alpha{alpha}"] = float(np.mean(accs))
        grid.append(row)
        print(f"n={n:5d}  a=0: {row['alpha0.0']:.4f}   a=1: {row['alpha1.0']:.4f}")
    res["alpha_by_n"] = grid

    # ---------- 3. 两个模型预测不一致的样本 ----------
    m_full = CategoricalNB(alpha=1.0).fit(Xtr, ytr)
    m_1k = CategoricalNB(alpha=1.0).fit(Xtr1, ytr1)
    pf, p1 = m_full.predict(Xte), m_1k.predict(Xte)
    res["disagreement"] = {
        "n_test": int(len(yte)),
        "n_differ": int((pf != p1).sum()),
        "frac_differ": float((pf != p1).mean()),
        "full_right_1k_wrong": int(np.sum((pf == yte) & (p1 != yte))),
        "full_wrong_1k_right": int(np.sum((pf != yte) & (p1 == yte))),
        "both_wrong": int(np.sum((pf != yte) & (p1 != yte))),
    }

    # ---------- 4. 过自信：平均最大后验 vs 准确率 ----------
    res["calibration"] = {
        "full": {
            "accuracy": accuracy(yte, pf),
            "mean_max_prob": float(m_full.predict_proba(Xte).max(axis=1).mean()),
            "frac_conf_above_099": float((m_full.predict_proba(Xte).max(axis=1) > 0.99).mean()),
            "acc_when_conf_above_099": float(
                (pf[m_full.predict_proba(Xte).max(axis=1) > 0.99] ==
                 yte[m_full.predict_proba(Xte).max(axis=1) > 0.99]).mean()),
        },
        "1000": {
            "accuracy": accuracy(yte, p1),
            "mean_max_prob": float(m_1k.predict_proba(Xte).max(axis=1).mean()),
        },
    }

    # ---------- 5. 特征数 m 与联合状态数 K ----------
    res["state_space"] = [
        {"m": m, "K": int(3 ** m), "params_full": int(7 * (3 ** m - 1)),
         "params_nb": int(7 * m * 2 + 6)}
        for m in (2, 4, 8, 12, 16)
    ]

    with open(os.path.join(OUT, "extra.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("\nzero cells (full):", res["zero_cells_full"]["zero_cells"], "/",
          res["zero_cells_full"]["total_cells"], res["zero_cells_full"]["zero_fraction"])
    print("zero cells (1000):", res["zero_cells_1000"]["zero_cells"], "/",
          res["zero_cells_1000"]["total_cells"])
    print("disagreement:", res["disagreement"])
    print("calibration:", res["calibration"])
    print("state space:", res["state_space"])


if __name__ == "__main__":
    main()
