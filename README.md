# 朴素贝叶斯模型与干豆分类探究

本项目对应“概率论与数理统计探究项目”的四个步骤：推广有限多值特征与多类别的朴素贝叶斯模型、比较两种训练规模、讨论连续特征与条件独立性，以及改进分类结果。

## 主要结果

所有测试准确率均在同一批 4085 条测试样本上计算，使用 `bean_raw.csv` 中 `partition == "TEST"` 的固定划分。

| 阶段 | 模型与输入 | 测试准确率 |
|---|---|---:|
| 第 2 步 | 离散 NB，训练集 2（1000 条） | 83.40% |
| 第 2 步 | 离散 NB，训练集 1（9526 条） | 83.87% |
| 第 3 步 | 高斯 NB，16 个连续特征 | 89.84% |
| 第 3 步 | KDE NB，16 个连续特征 | 90.04% |
| 第 4 步 | **标准化 RBF 核 SVC，27 个特征，`C=10`、`gamma="scale"`** | **93.29%** |

最终 SVC 由训练集上的五折分层交叉验证选择，CV 均值为 **93.04%**；测试集判对 **3811/4085** 个样本，比全量离散 NB 提高约 **9.42 个百分点**。16 维 SVC 的测试分数虽略高，最终模型仍按 CV 规则选择。

两个离散模型的准确率相差约 0.47 个百分点；配对 McNemar 精确检验 `p=0.01266`。全量模型有可检出的统计优势，但实际提升较小。论文同时说明了反复使用同一测试集进行探索性诊断的局限，不将 CV 与测试分数接近视为未过拟合的证明。

## 编译论文

从项目根目录执行，需要 TeX Live / MacTeX 提供 `latexmk` 与 XeLaTeX：

```bash
cd naive_bayes/latex
bash build.sh
```

输出为 `naive_bayes/output/pdf/朴素贝叶斯模型探究报告.pdf`，编译辅助文件位于 `latex/build/`。

使用 Overleaf 时，解压源码包并上传其中的 `latex/` 目录，选择 `main.tex` 为主文件、**XeLaTeX** 为编译器。源码已包含所需图表，使用 TeX Live 自带字体；仅编译论文无需 Python 或原始数据。

## 复现实验与核对结果

以下命令均在 `naive_bayes/` 目录执行：

```bash
python -m pip install -r requirements.txt

# 核对数据、朴素贝叶斯、配对统计量及最终 SVC
python latex/audit_report.py --verify-svc

# 重新生成完整实验结果（会覆盖对应的原始结果文件）
python code/run_step2.py
python code/run_step3.py
python code/run_extra.py
python code/run_step4.py

# 更新原稿 PNG 图与论文矢量图、表格
python code/make_figures.py
python latex/prepare_assets.py
```

复核脚本只新增或更新 `results/latex_audit.json`，不覆盖 `step2.json`、`step3.json`、`extra.json` 或 `step4.json`。省略 `--verify-svc` 可跳过 SVC 训练与交叉验证。论文图表再生成需要原实验结果及复核 JSON，以及可用的中文 matplotlib 字体。

本次已重跑两个规模的离散 NB、连续高斯 NB，以及最终 SVC 的五折 CV 和测试预测；其余候选、KDE 与重复特征实验采用保存结果并核对代码，未全部重新训练。

原实验环境为 Python 3.9、NumPy 2.0.2、pandas 2.2.3、scikit-learn 1.6.1、matplotlib 3.9.4。`requirements.txt` 只限定最低版本，不是环境锁定文件。第 2、3 步只依赖 NumPy 与 pandas，第 4 步及 SVC 复核需要 scikit-learn。

随机种子使用 `42`、`20261102` 或 `7`，具体见脚本。复核 CV 时须保留原始 CSV 行序；第 2 步为普通随机五折，第 4 步为分层五折。跨依赖版本及 SVC 内部概率估计不保证逐位一致。

## 数据与预处理

| 文件 | 样本数 | 说明 |
|---|---:|---|
| `bean_raw.csv` | 13611 | 20 列，包含 16 个连续特征、类别及划分信息 |
| `bean_train.csv` | 9526 | 离散训练集 1，特征取值为 0、1、2 |
| `bean_train_1000.csv` | 1000 | 训练集 1 的真子集 |
| `bean_test.csv` | 4085 | 固定离散测试集 |

数据文件保持原样。核对确认训练与测试 ID 无交集、原始与离散文件标签一致、没有缺失值。

用训练集的三分位顺序统计量作为切点并采用右闭区间，可以完全重现全部 16 个特征、13611 行的离散编码。例如 Area 的切点为 **38923、53381**；原稿关于“全体数据分位数分箱导致测试信息泄漏”的判断已修正。完整边界见 `results/latex_audit.json`。

## 代码与目录

| 路径 | 内容 |
|---|---|
| `code/nb_core.py` | 从零实现的离散与高斯 NB、评价指标 |
| `code/run_step2.py` | 两种训练规模、平滑系数、学习曲线、300 次子集抽样与模型比较 |
| `code/run_step3.py` | 高斯与 KDE NB、相关性、重复及删除特征实验 |
| `code/run_extra.py` | 空单元格、小样本平滑及过度自信分析 |
| `code/run_step4.py` | 21 个候选模型、分层 CV 选择及统一测试评估 |
| `code/make_figures.py` | 从保存结果生成八张原稿 PNG 图 |
| `results/` | 原始 JSON / NPY 结果及论文复核 JSON |
| `figures/` | 原稿使用的 PNG 图 |