# Supercon_Pred — 基于机器学习的超导转变温度（Tc）预测工具

模块化 Python 工具包，从化学组成出发，利用机器学习预测材料的超导转变温度（Tc）。专为材料发现流程设计，支持化学式特征工程、集成学习、高斯过程回归和贝叶斯超参数优化。

## 功能特点

- **自动特征生成** — 从化学式自动计算 45 维材料特征向量，涵盖原子半径、原子质量、价电子数、电负性、第一电离能、d 轨道特性等属性，每个属性包含 8 种统计描述符（均值、加权均值、标准差、加权标准差、极差、加权极差、熵、加权熵）。
- **三种 ML 模型** — 随机森林（Random Forest）、梯度提升（Gradient Boosting，均采用 `skopt` 贝叶斯超参搜索），以及高斯过程回归（Gaussian Process Regression，Rational Quadratic 核函数）。
- **全面评估体系** — R平方、MAE、MSE、RMSE 评分指标；特征相关系数热力图；内置特征重要性和排列重要性分析；预测值 vs 真实值散点图。
- **贝叶斯超参数优化** — 200 次迭代 x 10 折交叉验证，自动搜索最优模型参数。
- **模块化 CLI** — 三个子命令（`features`、`train`、`predict`），流程清晰。
- **模型持久化** — 训练好的模型和 StandardScaler 通过 pickle 序列化保存，可重复使用。

## 框架结构

```
Supercon_Pred/
├── __init__.py              # 包标记
├── config.py                # 集中配置（路径、超参空间、并行度）
├── utils.py                 # 工具函数（目录创建）
├── main.py                  # CLI 入口（argparse）
│
├── data/
│   ├── train/               # 训练数据目录
│   └── predict/             # 预测数据目录
│
├── features/
│   ├── atoms.py             # 元素属性字典（96 种元素）和化学式解析
│   └── generator.py         # 特征向量计算（45 维）
│
├── training/
│   ├── models.py            # ModelTrainer：RF / GB / GPR 训练 + BayesSearchCV
│   └── evaluator.py         # Evaluator：评分指标、特征重要性、可视化
│
├── prediction/
│   └── predictor.py         # Predictor：加载已训练模型预测 Tc
│
└── outputs/
    ├── models/              # 序列化模型文件 (.pickle)
    ├── figures/             # 评估图表（热力图、重要性、散点图）
    └── results/             # 评分结果和预测 CSV
```

## 实现方式

### 特征工程

通过正则表达式解析化学式，对每种元素从内置查找表（96 种元素）中提取以下属性：

- **原子半径**（埃）
- **原子质量**（amu）
- **价电子数**
- **电负性**（Pauling 标度）
- **第一电离能**（eV）
- **电子层配置**（来自 `chemlib`）

每种属性在全部组成元素上计算 8 种统计描述符：

| 描述符 | 未加权 | 化学计量比加权 |
|--------|--------|----------------|
| 均值 | `mean` | `mean_wtd` |
| 标准差 | `std` | `std_wtd` |
| 极差 | `range` | `range_wtd` |
| 熵 | `entropy` | `entropy_wtd` |

此外还有 4 个 d 轨道特征：壳层跨度、d 电子均值、d 轨道占比、d 轨道未填满数。

**合计：1（元素种类数）+ 5 x 8（统计特征）+ 4（d 轨道）= 45 维。**

### 模型训练

| 模型 | 算法 | 超参数优化 |
|------|------|-----------|
| **随机森林** | `sklearn.ensemble.RandomForestRegressor` | BayesSearchCV，200 次迭代，10 折 CV |
| **梯度提升** | `sklearn.ensemble.GradientBoostingRegressor` | BayesSearchCV，200 次迭代，10 折 CV |
| **高斯过程** | `sklearn.gaussian_process.GaussianProcessRegressor`，RationalQuadratic 核 | 固定核参数 |

数据按 90/10 划分训练/测试集，特征通过 `StandardScaler` 标准化。

### 评估指标

- 训练集和测试集的 R平方、MAE、MSE、RMSE
- 特征相关系数热力图
- 内置特征重要性（RF/GB）+ 排列重要性（所有模型）
- Top-6 特征 vs Tc 散点图
- 预测值 vs 真实值散点图

## 安装

### 环境要求

- Python 3.10+
- pip

### 安装步骤

```bash
git clone https://github.com/your-username/Supercon_Pred.git
cd Supercon_Pred
pip install -r requirements.txt
```

### 依赖项

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
scikit-optimize>=0.9
matplotlib>=3.4
seaborn>=0.11
chemlib>=0.1
```

## 使用方法

### 工作流程

```
[化合物 CSV] ---> features ---> [特征 CSV] ---> train ---> [模型 + 评估]
                                                   |
                                            predict ---> [预测结果 CSV]
```

### 1. 准备数据

创建一个包含 `formula` 列的 CSV 文件，写入化学式（如 `MgB2`、`YBa2Cu3O7`）。如果需要训练模型，请额外包含 `Tc` 列记录已知的超导转变温度。

**示例**（`data/train/materials.csv`）：

```csv
formula,Tc
MgB2,39.0
Nb3Sn,18.0
YBa2Cu3O7,93.0
```

**注意**：程序默认以 GBK 编码读取 CSV（中文 Windows 系统常见编码）。如果文件为 UTF-8 编码，程序会自动回退到 UTF-8。

### 2. 生成特征

```bash
python -m Supercon_Pred.main features data/train/materials.csv \
    -o data/train/materials_features.csv \
    --tc-col Tc
```

输出为 47 列 CSV（化学式 + 45 维特征 + Tc）。

### 3. 训练模型

```bash
python -m Supercon_Pred.main train data/train/materials_features.csv \
    --models RF,GB,GPR
```

- `--models`：逗号分隔的模型列表，可选 `RF`、`GB`、`GPR`，默认为全部三种。
- 训练时间取决于数据量。约 5000 条样本时，含贝叶斯优化约需 30 分钟。
- 训练好的模型自动保存至 `outputs/models/`。
- 评估图表保存至 `outputs/figures/`。
- 评分报告保存至 `outputs/results/`。

**可选**：训练完成后立即对新数据进行预测：

```bash
python -m Supercon_Pred.main train data/train/materials_features.csv \
    --models RF,GB,GPR \
    --predict data/predict/new_materials_features.csv
```

### 4. 预测 Tc

```bash
python -m Supercon_Pred.main predict data/predict/new_materials_features.csv \
    --model RandomForest \
    -o results/predictions.csv
```

模型名称可选：`RandomForest`、`GradientBoosting`、`GaussianProcess`。

### 5. 结果分析

**评分报告**（`outputs/results/model_scores.txt`）：

```
------ RandomForest -------
train: MAE=2.134, MSE=12.456, RMSE=3.529, R2=0.965
test:  MAE=5.891, MSE=78.234, RMSE=8.845, R2=0.813
----------------
```

**输出图表**（`outputs/figures/`）：

| 文件 | 说明 |
|------|------|
| `heatmap.png` | 特征相关系数热力图 |
| `feature_importance_*_builtin.png` | Top-10 内置特征重要性 |
| `feature_importance_*_permutation.png` | Top-10 排列重要性 |
| `scatter_*.png` | Top-6 特征 vs Tc 散点图 |
| `*_results_*.png` | 预测值 vs 真实值散点图（训练/测试） |

**预测结果**（`outputs/results/prediction_results_*.csv`）：

```csv
composition,Tc_prediction
MgB2,38.7
Nb3Sn,17.2
```

## 配置说明

`config.py` 中的关键配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `CPU_TOTAL` | `os.cpu_count()` | 可用 CPU 核心总数 |
| `ModelConfig.n_jobs_search_cv` | `CPU_TOTAL // 4` | 超参搜索并行任务数 |
| `RF_PARAM_SPACE` | `n_estimators: [20,200]` 等 | 随机森林搜索空间 |
| `GB_PARAM_SPACE` | `n_estimators: [20,500]` 等 | 梯度提升搜索空间 |

## 项目目录说明

- `Supercon_Pred/` — 主 Python 包
- `data/train/` — 训练数据集
- `data/predict/` — 预测数据集
- `outputs/` — 程序输出（模型、图表、结果）

## 引用说明

如果您在研究中使用了本程序，请引用以下论文：

> Xiaoying Li, et al., Machine learning accelerated search for superconductors in B-C-N based compounds and R3Ni2O7-type nickelates, *Physical Review B* **113**, 054521 (2026).

```bibtex
@article{Li2026SuperconPred,
  author  = {Xiaoying Li and others},
  title   = {Machine learning accelerated search for superconductors in {B-C-N} based compounds and {R3Ni2O7}-type nickelates},
  journal = {Phys. Rev. B},
  volume  = {113},
  pages   = {054521},
  year    = {2026}
}
```

## 许可证

本项目仅供学术研究使用。
