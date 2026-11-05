"""
run_step2.py
============
第 2 步：把推广后的（离散）朴素贝叶斯模型应用到 UCI Dry Bean 数据集。

输出 results/step2.json 以及若干中间结果，供报告引用。
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
    accuracy,
    classification_report,
    confusion_matrix,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261102)


def load(name):
    df = pd.read_csv(os.path.join(DATA, name))
    X = df.drop(columns=["id", "Class"]).to_numpy()
    y = df["Class"].to_numpy()
    return df, X, y


def evaluate(Xtr, ytr, Xte, yte, alpha=1.0):
    model = CategoricalNB(alpha=alpha).fit(Xtr, ytr)
    pred = model.predict(Xte)
    rep = classification_report(yte, pred)
    cm = confusion_matrix(yte, pred, labels=model.classes_)
    return model, pred, rep, cm


def main():
    res = {}
    tr_df, Xtr, ytr = load("bean_train.csv")
    tr1_df, Xtr1, ytr1 = load("bean_train_1000.csv")
    te_df, Xte, yte = load("bean_test.csv")
    labels = sorted(np.unique(ytr))

    res["data"] = {
        "n_train": int(len(ytr)),
        "n_train_1000": int(len(ytr1)),
        "n_test": int(len(yte)),
        "m_features": int(Xtr.shape[1]),
        "k_classes": int(len(labels)),
        "labels": labels,
        "k_i": [int(len(np.unique(Xtr[:, i]))) for i in range(Xtr.shape[1])],
        "train_class_counts": {c: int((ytr == c).sum()) for c in labels},
        "test_class_counts": {c: int((yte == c).sum()) for c in labels},
        "train_1000_class_counts": {c: int((ytr1 == c).sum()) for c in labels},
        "train_1000_is_subset": bool(set(tr1_df["id"]) <= set(tr_df["id"])),
    }

    # ---------- 0. 基线：多数类分类器 ----------
    maj = pd.Series(ytr).value_counts().idxmax()
    res["majority_baseline"] = {
        "class": str(maj),
        "accuracy": float(np.mean(yte == maj)),
    }

    # ---------- 1. 训练集 1（9526 条） ----------
    model1, pred1, rep1, cm1 = evaluate(Xtr, ytr, Xte, yte, alpha=1.0)
    res["train_full"] = {
        "accuracy": rep1["accuracy"],
        "report": rep1,
        "confusion_matrix": cm1.tolist(),
        "log_prior": {
            str(c): float(lp) for c, lp in zip(model1.classes_, model1.class_log_prior_)
        },
    }

    # ---------- 2. 训练集 2（1000 条） ----------
    model2, pred2, rep2, cm2 = evaluate(Xtr1, ytr1, Xte, yte, alpha=1.0)
    res["train_1000"] = {
        "accuracy": rep2["accuracy"],
        "report": rep2,
        "confusion_matrix": cm2.tolist(),
    }

    # ---------- 2b. 两个模型的配对比较：McNemar 检验 + 准确率差的置信区间 ----------
    from math import sqrt

    def mcnemar(y_true, pred_a, pred_b):
        a_ok, b_ok = pred_a == y_true, pred_b == y_true
        n01 = int(np.sum(a_ok & ~b_ok))  # A 对 B 错
        n10 = int(np.sum(~a_ok & b_ok))  # A 错 B 对
        # 连续性校正的 McNemar 统计量，服从 1 自由度卡方
        b, c = max(n01, n10), min(n01, n10)
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0
        # 卡方(1) 的生存函数 = erfc(sqrt(chi2/2))
        import math

        p = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0
        return {"n_a_right_b_wrong": n01, "n_a_wrong_b_right": n10,
                "chi2": float(chi2), "p_value": float(p)}

    acc_a, acc_b = rep1["accuracy"], rep2["accuracy"]
    n_test = len(yte)
    # 两个比例之差的近似 95% CI（配对情形用 McNemar 判别性更好，这里给出保守区间）
    se = sqrt((acc_a * (1 - acc_a) + acc_b * (1 - acc_b)) / n_test)
    res["comparison_full_vs_1000"] = {
        "accuracy_full": acc_a,
        "accuracy_1000": acc_b,
        "diff_pp": float((acc_a - acc_b) * 100),
        "diff_ci95_pp": [float((acc_a - acc_b - 1.96 * se) * 100),
                         float((acc_a - acc_b + 1.96 * se) * 100)],
        "mcnemar": mcnemar(yte, pred1, pred2),
        "per_class_delta": {
            c: float(rep1["per_class"][c]["recall"] - rep2["per_class"][c]["recall"])
            for c in rep1["per_class"]
        },
    }

    # 训练集自身的（重代入）准确率，用于观察过拟合
    res["resubstitution"] = {
        "train_full": float(np.mean(model1.predict(Xtr) == ytr)),
        "train_1000": float(np.mean(model2.predict(Xtr1) == ytr1)),
    }

    # ---------- 3. 平滑系数 alpha 的影响（训练集内部 5 折交叉验证） ----------
    def cv_accuracy(X, y, alpha, folds=5, seed=0):
        rs = np.random.default_rng(seed)
        idx = rs.permutation(len(y))
        chunks = np.array_split(idx, folds)
        accs = []
        for f in range(folds):
            va = chunks[f]
            tr = np.concatenate([chunks[g] for g in range(folds) if g != f])
            mdl = CategoricalNB(alpha=alpha).fit(X[tr], y[tr])
            accs.append(accuracy(y[va], mdl.predict(X[va])))
        return float(np.mean(accs)), float(np.std(accs)), accs

    alphas = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    cv_full = [cv_accuracy(Xtr, ytr, a, seed=7) for a in alphas]
    cv_1k = [cv_accuracy(Xtr1, ytr1, a, seed=7) for a in alphas]
    # 直接在测试集上观察（仅作诊断，不用于挑选模型）
    test_acc = []
    for a in alphas:
        m = CategoricalNB(alpha=a).fit(Xtr, ytr)
        test_acc.append(accuracy(yte, m.predict(Xte)))
    res["alpha_sweep"] = {
        "alphas": alphas,
        "cv5_train_full_mean": [c[0] for c in cv_full],
        "cv5_train_full_std": [c[1] for c in cv_full],
        "cv5_train_1000_mean": [c[0] for c in cv_1k],
        "cv5_train_1000_std": [c[1] for c in cv_1k],
        "test_train_full": test_acc,
    }

    # ---------- 4. 学习曲线：准确率 ~ 训练样本量 ----------
    sizes = [50, 100, 200, 500, 1000, 2000, 4000, 6000, 8000, 9526]
    curve = []
    for n in sizes:
        accs = []
        reps = 30 if n <= 2000 else 10
        for r in range(reps):
            if n >= len(ytr):
                sub = np.arange(len(ytr))
            else:
                sub = RNG.choice(len(ytr), size=n, replace=False)
            mdl = CategoricalNB(alpha=1.0).fit(Xtr[sub], ytr[sub])
            accs.append(accuracy(yte, mdl.predict(Xte)))
        curve.append({"n": n, "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                      "reps": reps})
    res["learning_curve"] = curve

    # ---------- 5. 1000 条随机训练子集的准确率分布 ----------
    dist = []
    for r in range(300):
        sub = RNG.choice(len(ytr), size=1000, replace=False)
        mdl = CategoricalNB(alpha=1.0).fit(Xtr[sub], ytr[sub])
        dist.append(accuracy(yte, mdl.predict(Xte)))
    dist = np.array(dist)
    np.save(os.path.join(OUT, "dist_1000.npy"), dist)
    res["dist_1000"] = {
        "n_repeats": len(dist),
        "mean": float(dist.mean()),
        "std": float(dist.std()),
        "q025": float(np.quantile(dist, 0.025)),
        "q25": float(np.quantile(dist, 0.25)),
        "q50": float(np.quantile(dist, 0.50)),
        "q75": float(np.quantile(dist, 0.75)),
        "q975": float(np.quantile(dist, 0.975)),
        "given_1000": res["train_1000"]["accuracy"],
        "pct_rank_of_given": float((dist < res["train_1000"]["accuracy"]).mean()),
    }

    # ---------- 6. 单特征判别力（互信息 + 单特征 NB 准确率） ----------
    feat_names = [c for c in tr_df.columns if c not in ("id", "Class")]
    mi = []
    for i, name in enumerate(feat_names):
        # 互信息 I(X_i; Y)
        ct = pd.crosstab(tr_df[name], tr_df["Class"]).to_numpy().astype(float)
        pxy = ct / ct.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = pxy * np.log(pxy / (px * py))
        mi.append(float(np.nansum(term)))
        # 仅用该特征的 NB 准确率
        mdl = CategoricalNB(alpha=1.0).fit(Xtr[:, [i]], ytr)
        mi[-1] = {"feature": name, "mutual_info": mi[-1],
                  "single_feature_acc": accuracy(yte, mdl.predict(Xte[:, [i]]))}
    res["feature_info"] = mi

    # ---------- 7. 训练集 2 的构成 ----------
    res["train_1000_composition"] = {
        "class_counts": {c: int((ytr1 == c).sum()) for c in labels},
        "expected_if_random": {c: float(len(ytr1) * (ytr == c).mean()) for c in labels},
    }

    with open(os.path.join(OUT, "step2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(OUT, "cm_full.npy"), cm1)
    np.save(os.path.join(OUT, "cm_1000.npy"), cm2)

    print("majority baseline :", res["majority_baseline"]["accuracy"])
    print("NB(alpha=1) full  :", rep1["accuracy"])
    print("NB(alpha=1) 1000  :", rep2["accuracy"])
    print("resubstitution    :", res["resubstitution"])
    print("dist_1000         :", {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in res["dist_1000"].items()})
    print("alpha(cv full)    :", [round(x, 4) for x in res["alpha_sweep"]["cv5_train_full_mean"]])
    print("alpha(cv 1000)    :", [round(x, 4) for x in res["alpha_sweep"]["cv5_train_1000_mean"]])
    print("alpha(test)       :", [round(x, 4) for x in test_acc])
    print("learning curve    :", [(c["n"], round(c["mean"], 4)) for c in curve])
    print("\nper-class (full):")
    for k, v in rep1["per_class"].items():
        print(f"  {k:10s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"n={v['support']}")
    print("\nmacro", {k: round(v, 4) for k, v in rep1["macro"].items()})
    print("weighted", {k: round(v, 4) for k, v in rep1["weighted"].items()})
    print("\nfeatures:", [(f["feature"], round(f["mutual_info"], 3),
                           round(f["single_feature_acc"], 3)) for f in mi])


if __name__ == "__main__":
    main()
