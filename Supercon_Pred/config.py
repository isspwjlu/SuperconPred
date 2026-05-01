"""集中管理所有配置参数。"""
import os
from dataclasses import dataclass
from skopt.space import Integer, Real


# ---------- 路径配置 ----------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
PREDICT_DATA_DIR = os.path.join(DATA_DIR, "predict")

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")


# ---------- 全局配置 ----------
CPU_TOTAL: int = os.cpu_count() or 4
# ---------- 模型并行配置 ----------
@dataclass
class ModelConfig:
    enable_random_forest: bool = True
    enable_gradient_boosting: bool = True
    enable_gaussian_process: bool = True
    n_jobs_search_cv: int = max(1, CPU_TOTAL // 4)
    n_jobs_model: int = 4


# ---------- 超参搜索空间 ----------
# NOTE: 与原版程序相比，搜索参数使用 skopt 的 (low, high) 连续范围替代了 np.arange 离散步进。
# 这使得搜索更精细、结果略好，但与原始结果不完全一致。如需完全复现原结果，
# 可将参数改为: "n_estimators": (20, 200, 10) 等步进格式。
RF_PARAM_SPACE = {
    "n_estimators": Integer(20, 200),
    "max_depth": Integer(1, 12),
    "max_features": Integer(8, 40),
}

GB_PARAM_SPACE = {
    "n_estimators": Integer(20, 500),
    "max_depth": Integer(1, 12),
    "learning_rate": Real(0.01, 0.1),
    "min_samples_leaf": Integer(1, 8),
    "alpha": Real(0.1, 0.9),
}

# ---------- 其他 ----------
FIGS_FOR_FEATURE_IMPORTANCE: int = 6
CORRELATION_HEATMAP_FIGSIZE: tuple = (58, 48)
