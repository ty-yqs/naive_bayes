"""
run_step4.py
============
第 4 步：在原始连续特征上做改进，尽可能提高测试集分类准确率。

原则（避免过拟合测试集）：
  * 所有模型选择、超参数选择都只用训练集上的 5 折分层交叉验证（CV）完成；
  * 测试集（partition == TEST，4085 条）只在最后评估一次；
  * 预处理（标准化、特征构造）都放在 Pipeline 里，只在训练折上 fit，避免信息泄漏。

依赖：numpy, pandas, scikit-learn
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
SEED = 42
NFOLD = 5


# ---------------------------------------------------------------------------
# 特征工程：只用“单个样本内部”的比值，不依赖训练集统计量
# ---------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    A, P = df["Area"], df["Perimeter"]
    maj, mino = df["MajorAxisLength"], df["MinorAxisLength"]
    conv, eqd = df["ConvexArea"], df["EquivDiameter"]
    out = df.copy()
    out["solidity2"] = A / conv                     # 与 Solidity 同义
    out["roundness2"] = 4 * np.pi * A / (P ** 2)    # 与 roundness 同义
    out["compactness2"] = P ** 2 / (4 * np.pi * A)
    out["aspect2"] = maj / mino                     # 与 AspectRation 同义
    out["major_per_P"] = maj / P
    out["minor_per_P"] = mino / P
    out["eqd_per_P"] = eqd / P
    out["area_per_majmin"] = A / (maj * mino)
    out["sf3_alt"] = (conv - A) / conv
    out["log_area"] = np.log(A)
    out["log_perimeter"] = np.log(P)
    return out


BASE = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
        "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
        "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3",
        "ShapeFactor4"]

ENG = ["solidity2", "roundness2", "compactness2", "aspect2", "major_per_P",
       "minor_per_P", "eqd_per_P", "area_per_majmin", "sf3_alt", "log_area",
       "log_perimeter"]

# 与其余特征高度共线、可删除的一组特征（消融用）
REDUNDANT = ["Area", "ConvexArea", "EquivDiameter", "AspectRation", "Compactness"]
LEAN = [c for c in BASE if c not in REDUNDANT]


def pipe(model, scaler=True):
    steps = []
    if scaler:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps)


def main():
    t00 = time.time()
    raw = pd.read_csv(os.path.join(ROOT, "bean_raw.csv"))
    tr = engineer(raw[raw.partition == "TRAIN"].reset_index(drop=True))
    te = engineer(raw[raw.partition == "TEST"].reset_index(drop=True))
    ytr, yte = tr["Class"].to_numpy(), te["Class"].to_numpy()
    res = {"data": {"n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "classes": sorted(np.unique(ytr).tolist())}}
    cv = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

    rows = []
    fitted = {}

    def add(name, model, feats, family):
        t0 = time.time()
        X = tr[feats].to_numpy(float)
        s = cross_val_score(pipe(model), X, ytr, cv=cv, scoring="accuracy", n_jobs=1)
        p = pipe(model).fit(X, ytr)
        fitted[name] = (p, feats)
        rows.append({"model": name, "family": family, "n_features": len(feats),
                     "cv_mean": float(s.mean()), "cv_std": float(s.std()),
                     "seconds": round(time.time() - t0, 1)})
        print(f"{name:16s} CV = {s.mean():.4f} ± {s.std():.4f}  ({len(feats)} feats)")
        return s.mean()

    # ------------------------------------------------------------------
    # 阶段 A：不同模型族在原始 16 个特征上的比较
    # ------------------------------------------------------------------
    print("=== 阶段 A：模型族比较（16 个原始特征，标准化） ===")
    add("GaussianNB", SkGaussianNB(), BASE, "生成式/贝叶斯")
    add("LDA", LinearDiscriminantAnalysis(), BASE, "线性判别")
    add("QDA", QuadraticDiscriminantAnalysis(reg_param=0.05), BASE, "二次判别")
    add("kNN_k15", KNeighborsClassifier(n_neighbors=15, weights="distance"), BASE, "近邻")
    add("LogisticReg", LogisticRegression(C=1.0, max_iter=3000), BASE, "线性判别")
    add("SVC_rbf", SVC(C=10, gamma="scale", probability=True, cache_size=800), BASE, "核方法")
    add("RandomForest", RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                               random_state=SEED), BASE, "树集成")
    add("ExtraTrees", ExtraTreesClassifier(n_estimators=500, n_jobs=-1,
                                           random_state=SEED), BASE, "树集成")
    add("HistGB", HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=SEED), BASE, "树集成")

    # ------------------------------------------------------------------
    # 阶段 B：特征集消融（特征工程 / 冗余特征删除）
    # ------------------------------------------------------------------
    print("\n=== 阶段 B：特征集消融 ===")
    add("RandomForest_ENG", RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                                   random_state=SEED), BASE + ENG, "树集成")
    add("ExtraTrees_ENG", ExtraTreesClassifier(n_estimators=500, n_jobs=-1,
                                               random_state=SEED), BASE + ENG, "树集成")
    add("HistGB_ENG", HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=SEED), BASE + ENG, "树集成")
    add("SVC_ENG", SVC(C=10, gamma="scale", probability=True, cache_size=800),
        BASE + ENG, "核方法")
    add("SVC_LEAN", SVC(C=10, gamma="scale", probability=True, cache_size=800),
        LEAN, "核方法")

    # ------------------------------------------------------------------
    # 阶段 C：超参数微调
    # ------------------------------------------------------------------
    print("\n=== 阶段 C：超参数微调 ===")
    for C in (1.0, 100.0):
        add(f"SVC_C{int(C)}", SVC(C=C, gamma="scale", probability=True,
                                  cache_size=800), BASE, "核方法")
    for k in (5, 9, 25):
        add(f"kNN_k{k}", KNeighborsClassifier(n_neighbors=k, weights="distance"),
            BASE, "近邻")
    add("HistGB_leaf63", HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=63, l2_regularization=0.5,
        early_stopping=False, random_state=SEED), BASE + ENG, "树集成")

    # ------------------------------------------------------------------
    # 阶段 D：软投票集成
    # ------------------------------------------------------------------
    print("\n=== 阶段 D：软投票集成 ===")
    members = ["SVC_rbf", "RandomForest_ENG", "ExtraTrees_ENG", "HistGB_leaf63",
               "LogisticReg"]
    union = []
    for m in members:
        for c in fitted[m][1]:
            if c not in union:
                union.append(c)
    ests = [(m, pipe(clone(fitted[m][0].named_steps["clf"]))) for m in members]
    vote = VotingClassifier(estimators=ests, voting="soft", n_jobs=-1)
    X = tr[union].to_numpy(float)
    s = cross_val_score(vote, X, ytr, cv=cv, scoring="accuracy", n_jobs=1)
    vote.fit(X, ytr)
    rows.append({"model": "SoftVoting", "family": "集成", "n_features": len(union),
                 "cv_mean": float(s.mean()), "cv_std": float(s.std()),
                 "seconds": 0.0})
    print(f"{'SoftVoting':16s} CV = {s.mean():.4f} ± {s.std():.4f}  ({len(union)} feats)")

    # ------------------------------------------------------------------
    # 测试集评估：每个候选模型各评估一次
    # ------------------------------------------------------------------
    print("\n=== 最终测试集评估 ===")
    test_rows = []
    preds = {}
    for r in rows:
        name = r["model"]
        if name == "SoftVoting":
            p, feats, Xte = vote, union, te[union].to_numpy(float)
        else:
            p, feats = fitted[name]
            Xte = te[feats].to_numpy(float)
        pred = p.predict(Xte)
        preds[name] = pred
        acc = float(accuracy_score(yte, pred))
        test_rows.append({"model": name, "family": r["family"],
                          "cv_mean": r["cv_mean"], "test_acc": acc,
                          "gap_pp": round((r["cv_mean"] - acc) * 100, 3),
                          "n_features": r["n_features"],
                          "seconds": r["seconds"]})
        print(f"  {name:16s} test = {acc:.4f}   (CV {r['cv_mean']:.4f}, "
              f"gap {(r['cv_mean']-acc)*100:+.2f} pp)")

    res["cv_table"] = rows
    res["test_table"] = test_rows

    best_name = max(rows, key=lambda r: r["cv_mean"])["model"]
    if best_name == "SoftVoting":
        p, feats, Xte = vote, union, te[union].to_numpy(float)
    else:
        p, feats = fitted[best_name]
        Xte = te[feats].to_numpy(float)
    pred_best = p.predict(Xte)
    labels = sorted(np.unique(ytr))
    rep = classification_report(yte, pred_best, labels=labels, output_dict=True,
                                digits=4)
    cm = confusion_matrix(yte, pred_best, labels=labels)
    proba = p.predict_proba(Xte)
    res["best"] = {
        "name": best_name,
        "features": list(feats),
        "cv_mean": [r["cv_mean"] for r in rows if r["model"] == best_name][0],
        "test_acc": float(accuracy_score(yte, pred_best)),
        "report": rep,
        "confusion_matrix": cm.tolist(),
        "labels": [str(x) for x in labels],
        "mean_max_prob": float(proba.max(axis=1).mean()),
        "log_loss": float(-np.mean(np.log(np.clip(
            proba[np.arange(len(yte)), [labels.index(v) for v in yte]], 1e-15, 1)))),
    }
    # 次优模型：用于稳健性讨论
    order = sorted(rows, key=lambda r: -r["cv_mean"])
    second = order[1]["model"]
    res["second_best"] = {
        "name": second,
        "test_acc": float(accuracy_score(yte, preds[second])),
        "agreement_with_best": float(np.mean(preds[second] == pred_best)),
    }

    # 特征重要性（RF_ENG）
    rf = fitted["RandomForest_ENG"][0].named_steps["clf"]
    imp = sorted(zip(fitted["RandomForest_ENG"][1], rf.feature_importances_),
                 key=lambda t: -t[1])
    res["rf_importance"] = [{"feature": f, "importance": float(v)} for f, v in imp]

    res["elapsed_seconds"] = round(time.time() - t00, 1)
    with open(os.path.join(OUT, "step4.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(OUT, "cm_best.npy"), cm)

    print("\nBEST:", best_name, "test =", round(res["best"]["test_acc"], 4))
    print("elapsed", res["elapsed_seconds"], "s")


if __name__ == "__main__":
    main()
