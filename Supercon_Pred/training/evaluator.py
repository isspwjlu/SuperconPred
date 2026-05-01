"""模型评估与可视化。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

from .. import config
from ..utils import ensure_dir


class Evaluator:
    """模型评估器：相关系数热力图、特征重要性、预测散点图、评分指标。"""

    def __init__(self, df: pd.DataFrame, feature_names: list[str]):
        self.df = df
        self.feature_names = feature_names

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
        """绘制特征相关系数热力图。"""
        fig, ax = plt.subplots(figsize=config.CORRELATION_HEATMAP_FIGSIZE)
        plt.subplots_adjust(left=0.15, right=1, bottom=0.15, top=0.9)
        sns.heatmap(df.corr(method='pearson'), annot=False, fmt=".2f", cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap', fontsize=60)
        plt.xticks(rotation=45, fontsize=35)
        plt.yticks(fontsize=35)
        ensure_dir(config.FIGURE_DIR)
        plt.savefig(save_path)
        plt.close()
        print(f"[绘图] 热力图: {save_path}")

    def plot_feature_importance(self, model, X_train, y_train, model_name: str):
        """绘制特征重要性（内置重要性 + 排列重要性 + 散点图）。"""
        if hasattr(model, 'feature_importances_'):
            fi = pd.Series(model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
            self._bar_plot(fi.head(10), f'{model_name} - Builtin Importance',
                           f"{config.FIGURE_DIR}/feature_importance_{model_name}_builtin.png")

        perm = permutation_importance(model, X_train, y_train, n_repeats=20)
        idx = perm.importances_mean.argsort()[::-1]
        perm_s = pd.Series(perm.importances_mean[idx], index=np.array(self.feature_names)[idx])
        self._bar_plot(perm_s.head(10), f'{model_name} - Permutation Importance',
                       f"{config.FIGURE_DIR}/feature_importance_{model_name}_permutation.png")

        self._scatter_plot(idx[:config.FIGS_FOR_FEATURE_IMPORTANCE], model_name)

    def _bar_plot(self, series: pd.Series, title: str, save_path: str):
        """保存条形图。"""
        fig, ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9)
        sns.barplot(x=series.values, y=series.index, ax=ax)
        plt.title(title)
        ensure_dir(config.FIGURE_DIR)
        plt.savefig(save_path)
        plt.close()

    def _scatter_plot(self, indices: list, model_name: str):
        """绘制 top-N 特征 vs Tc 散点图。"""
        n = len(indices)
        fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(35, 25))
        axes = axes.flatten()
        for i, idx in enumerate(indices):
            axes[i].scatter(self.df.iloc[:, idx], self.df.iloc[:, -1], s=200, c='b', alpha=0.7)
            axes[i].set_xlabel(self.feature_names[idx], fontsize=26)
            axes[i].set_ylabel('Tc', fontsize=26)
            axes[i].tick_params(labelsize=20)
        plt.suptitle(f'{model_name} - Feature Scatter', fontsize=36)
        ensure_dir(config.FIGURE_DIR)
        plt.savefig(f"{config.FIGURE_DIR}/scatter_{model_name}.png")
        plt.close()

    @staticmethod
    def save_scores(y_train_pred: dict, y_test_pred: dict,
                    y_train_true: np.ndarray, y_test_true: np.ndarray, save_path: str):
        """计算并保存所有模型的评分指标。"""
        ensure_dir(config.RESULT_DIR)
        with open(save_path, 'w') as f:
            for name in y_train_pred:
                f.write(f'------ {name} -------\n')
                for split, preds, true in [('train', y_train_pred[name], y_train_true),
                                           ('test', y_test_pred[name], y_test_true)]:
                    mae = mean_absolute_error(true, preds)
                    mse = mean_squared_error(true, preds)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(true, preds)
                    f.write(f'{split}: MAE={mae:.3f}, MSE={mse:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}\n')
                f.write('----------------\n')
        print(f"[评分] 已保存: {save_path}")

    @staticmethod
    def plot_prediction_scatter(y_true, y_pred, model_name: str, split: str):
        """绘制预测值 vs 真实值散点图并保存预测结果CSV。"""
        fig = plt.figure()
        plt.scatter(y_true, y_pred, s=30, marker='o', facecolors='none',
                    edgecolors='blue' if split == 'train' else 'red', label=f'{split}')
        plt.plot(y_true, y_true, '-k', linewidth=1)
        plt.xlabel('True Tc')
        plt.ylabel('Predicted Tc')
        plt.title(f'{model_name} - {split}')
        pd.DataFrame({f'y_{split}': y_true, 'predicted': y_pred}).to_csv(
            f"{config.RESULT_DIR}/y_{split}_{model_name}.csv", index=False)
        plt.savefig(f"{config.FIGURE_DIR}/{split}_results_{model_name}.png")
        plt.close()
