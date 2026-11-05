"""
make_figures.py
===============
把 results/*.json 中的数值画成报告用的图，输出到 figures/。
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.axisbelow"] = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

C1, C2, C3 = "#2b5d8c", "#c0504d", "#4f8f6b"


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
def fig_learning_curve(s2):
    lc = s2["learning_curve"]
    n = np.array([r["n"] for r in lc], float)
    m = np.array([r["mean"] for r in lc]) * 100
    sd = np.array([r["std"] for r in lc]) * 100
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(n, m, yerr=sd, marker="o", ms=4, lw=1.6, capsize=3, color=C1,
                label="不同随机子集上的均值 ± 标准差")
    ax.axhline(s2["train_full"]["accuracy"] * 100, ls="--", lw=1.2, color=C2,
               label=f"全部训练集 (N=9526)：{s2['train_full']['accuracy']*100:.2f}%")
    ax.axhline(s2["train_1000"]["accuracy"] * 100, ls=":", lw=1.2, color=C3,
               label=f"训练集 2 (N=1000)：{s2['train_1000']['accuracy']*100:.2f}%")
    ax.set_xlabel("训练样本数 N")
    ax.set_ylabel("测试集分类准确率 (%)")
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([int(v) for v in n], rotation=30, fontsize=8)
    ax.set_ylim(74, 86)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("离散朴素贝叶斯的学习曲线")
    fig.savefig(os.path.join(FIG, "fig_learning_curve.png"))
    plt.close(fig)


def fig_dist1000(s2):
    d = s2["dist_1000"]
    dist = np.load(os.path.join(RES, "dist_1000.npy")) * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(dist, bins=22, color=C1, alpha=0.75, edgecolor="white",
            label=f"300 个随机 1000 样本子集（均值 {dist.mean():.2f}%，"
                  f"标准差 {dist.std():.2f} pp）")
    ax.axvline(d["given_1000"] * 100, color=C3, lw=1.8,
               label=f"给定的训练集 2：{d['given_1000']*100:.2f}%")
    ax.axvline(s2["train_full"]["accuracy"] * 100, color=C2, ls="--", lw=1.5,
               label=f"全部训练集：{s2['train_full']['accuracy']*100:.2f}%")
    ax.set_xlabel("测试集分类准确率 (%)")
    ax.set_ylabel("频数")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("1000 个训练样本时的准确率波动（抽样 300 次）")
    fig.savefig(os.path.join(FIG, "fig_dist1000.png"))
    plt.close(fig)


def fig_confusion(s2):
    labels = s2["train_full"]["report"]["labels"]
    cms = [np.array(s2["train_full"]["confusion_matrix"], float),
           np.array(s2["train_1000"]["confusion_matrix"], float)]
    titles = [f"(a) 训练集 1（N=9526），准确率 {s2['train_full']['accuracy']*100:.2f}%",
              f"(b) 训练集 2（N=1000），准确率 {s2['train_1000']['accuracy']*100:.2f}%"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, cm, t in zip(axes, cms, titles):
        cmn = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cmn[i, j]:.0f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if cmn[i, j] > 55 else "black")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="行归一化百分比 (%)")
    fig.savefig(os.path.join(FIG, "fig_confusion.png"))
    plt.close(fig)


def fig_corr(s3):
    feats = s3["correlation"]["features"]
    C = np.array(s3["correlation"]["matrix"])
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [f.replace("MajorAxisLength", "Major").replace("MinorAxisLength", "Minor")
             .replace("ShapeFactor", "SF").replace("EquivDiameter", "EquivD")
             .replace("AspectRation", "Aspect") for f in feats]
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(len(feats)):
        for j in range(len(feats)):
            if i != j and abs(C[i, j]) > 0.95:
                ax.text(j, i, f"{C[i, j]:.3f}", ha="center", va="center",
                        fontsize=5.6, color="black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson 相关系数")
    ax.set_title("16 个形状特征的相关性矩阵（|r|>0.95 的格子标注数值）")
    fig.savefig(os.path.join(FIG, "fig_corr.png"))
    plt.close(fig)


def fig_dup(s3):
    a = s3["duplicate_area"]
    b = s3["duplicate_area_equivdiameter"]
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for axx, data, lab, col in [
        (ax[0], a, "只重复 Area", C1),
        (ax[1], b, "同时重复 Area 与 EquivDiameter", C2),
    ]:
        x = [d["reps"] for d in data]
        y = [d["accuracy"] * 100 for d in data]
        axx.plot(x, y, marker="o", ms=4.5, lw=1.5, color=col)
        for xi, yi in zip(x, y):
            axx.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=7)
        axx.set_xlabel("特征被重复的份数（1 = 不重复）")
        axx.set_ylabel("测试集准确率 (%)")
        axx.set_title(lab, fontsize=10)
        axx.set_xticks(x)
    fig.suptitle("重复使用相关特征对朴素贝叶斯准确率的影响（离散特征）", fontsize=11)
    fig.savefig(os.path.join(FIG, "fig_duplicate.png"))
    plt.close(fig)


def fig_alpha(s2):
    a = s2["alpha_sweep"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.arange(len(a["alphas"]))
    ax.plot(x, np.array(a["cv5_train_full_mean"]) * 100, marker="o", ms=4, color=C1,
            label="训练集 1 的 5 折 CV")
    ax.plot(x, np.array(a["cv5_train_1000_mean"]) * 100, marker="s", ms=4, color=C2,
            label="训练集 2 的 5 折 CV")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in a["alphas"]])
    ax.set_xlabel("平滑系数 α")
    ax.set_ylabel("交叉验证准确率 (%)")
    ax.set_title("拉普拉斯平滑系数的影响")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_alpha.png"))
    plt.close(fig)


def fig_class(s2, s3, s4=None):
    labels = s2["train_full"]["report"]["labels"]
    series = {
        "离散 NB（训练集 1）": [s2["train_full"]["report"]["per_class"][c]["f1"] for c in labels],
        "高斯 NB（连续特征）": [s3["gaussian_nb"]["report"]["per_class"][c]["f1"] for c in labels],
    }
    colors = [C1, C2]
    if s4 is not None:
        series["改进模型"] = [s4["best"]["report"][c]["f1-score"] for c in labels]
        colors.append(C3)
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    x = np.arange(len(labels))
    for k, (name, vals) in enumerate(series.items()):
        ax.bar(x + (k - (n - 1) / 2) * w, np.array(vals) * 100, width=w * 0.92,
               label=name, color=colors[k])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("各类 F1 分数 (%)")
    ax.set_ylim(40, 102)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.set_title("分品类 F1 分数对比", pad=26)
    fig.savefig(os.path.join(FIG, "fig_class_f1.png"))
    plt.close(fig)


def fig_importance(s4):
    imp = s4["rf_importance"][:12]
    names = [d["feature"] for d in imp][::-1]
    vals = [d["importance"] for d in imp][::-1]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.barh(names, vals, color=C1)
    ax.set_xlabel("随机森林特征重要性（不纯度下降）")
    ax.set_title("改进模型的特征重要性（前 12）")
    fig.savefig(os.path.join(FIG, "fig_importance.png"))
    plt.close(fig)


def main():
    s2 = load("step2.json")
    s3 = load("step3.json")
    s4 = load("step4.json") if os.path.exists(os.path.join(RES, "step4.json")) else None
    fig_learning_curve(s2)
    fig_dist1000(s2)
    fig_confusion(s2)
    fig_corr(s3)
    fig_dup(s3)
    fig_alpha(s2)
    fig_class(s2, s3, s4)
    if s4 is not None:
        fig_importance(s4)
    print("figures written to", FIG)
    print(sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
