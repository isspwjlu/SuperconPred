"""模型训练与超参优化。"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

import config
from utils import ensure_dir


class ModelTrainer:
    """统一模型训练器，支持 RandomForest / GradientBoosting / GaussianProcess。"""

    def __init__(self, data_file: str,
                 enable_rf: bool = True,
                 enable_gb: bool = True,
                 enable_gpr: bool = True):
        self.data_file = data_file
        self.enable_rf = enable_rf
        self.enable_gb = enable_gb
        self.enable_gpr = enable_gpr
        self.scaler = StandardScaler()
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.models: dict[str, object] = {}
        self.y_train_pred: dict[str, np.ndarray] = {}
        self.y_test_pred: dict[str, np.ndarray] = {}
        self.feature_names: list[str] = []

    def load_data(self):
        """加载数据、划分训练/测试集、归一化。"""
        try:
            df = pd.read_csv(self.data_file, header=0, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(self.data_file, header=0, encoding='utf-8')
        n_before = len(df)
        df = df.dropna()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[数据] 已删除 {n_dropped} 行含缺失值的数据")
        if df.empty:
            raise ValueError(f"数据为空（所有行均含缺失值）: {self.data_file}")
        if df.dtypes[df.keys()[0]] == 'object':
            df.drop(df.columns[0], axis=1, inplace=True)
        self.feature_names = list(df.columns[:-1])
        data = df.values.astype(float)
        X, y = data[:, :-1], data[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=0.9, test_size=0.1, random_state=42)
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print(f"[数据] 已加载: {df.shape[0]} 样本, {len(self.feature_names)} 特征")

    def _bayes_search(self, estimator, param_space: dict, n_iter: int = 200):
        """执行贝叶斯超参搜索。"""
        search = BayesSearchCV(estimator, param_space, n_iter=n_iter, cv=10,
                               n_jobs=config.ModelConfig.n_jobs_search_cv,
                               scoring='r2', verbose=0)
        search.fit(self.X_train, self.y_train)
        return search

    def train_random_forest(self):
        """训练 RandomForestRegressor + 超参优化。"""
        print("[RF] 开始超参优化...")
        model = RandomForestRegressor(n_jobs=config.ModelConfig.n_jobs_model)
        search = self._bayes_search(model, config.RF_PARAM_SPACE)
        best = search.best_estimator_
        final = RandomForestRegressor(**search.best_params_, n_jobs=config.ModelConfig.n_jobs_model)
        final.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = final
        self.y_train_pred['RandomForest'] = final.predict(self.X_train)
        self.y_test_pred['RandomForest'] = final.predict(self.X_test)
        print(f"[RF] 最佳参数: {search.best_params_}")
        print(f"[RF] 最佳评分: {search.best_score_:.3f}")
        return best, final

    def train_gradient_boosting(self):
        """训练 GradientBoostingRegressor + 超参优化。"""
        print("[GB] 开始超参优化...")
        model = GradientBoostingRegressor()
        search = self._bayes_search(model, config.GB_PARAM_SPACE)
        best = search.best_estimator_
        final = GradientBoostingRegressor(**search.best_params_)
        final.fit(self.X_train, self.y_train)
        self.models['GradientBoosting'] = final
        self.y_train_pred['GradientBoosting'] = final.predict(self.X_train)
        self.y_test_pred['GradientBoosting'] = final.predict(self.X_test)
        print(f"[GB] 最佳参数: {search.best_params_}")
        print(f"[GB] 最佳评分: {search.best_score_:.3f}")
        return best, final

    def train_gaussian_process(self):
        """训练 GaussianProcessRegressor。"""
        print("[GPR] 开始训练...")
        kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
        model = GaussianProcessRegressor(kernel=kernel, alpha=0.5)
        model.fit(self.X_train, self.y_train)
        self.models['GaussianProcess'] = model
        self.y_train_pred['GaussianProcess'] = model.predict(self.X_train)
        self.y_test_pred['GaussianProcess'] = model.predict(self.X_test)
        print("[GPR] 训练完成")
        return model

    def train_all(self) -> dict[str, object]:
        """按配置训练所有启用的模型并保存。"""
        ensure_dir(config.MODEL_DIR)
        self.load_data()
        rf_final = gb_final = None

        if self.enable_rf:
            _, rf_final = self.train_random_forest()
        if self.enable_gb:
            _, gb_final = self.train_gradient_boosting()
        if self.enable_gpr:
            self.train_gaussian_process()

        # 保存模型和归一化器
        for name, model in self.models.items():
            path = f"{config.MODEL_DIR}/{name}_model.pickle"
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"[模型] 已保存: {path}")

        scaler_path = f"{config.MODEL_DIR}/scaler.pickle"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"[模型] 归一化器已保存: {scaler_path}")
        return {'rf_final': rf_final, 'gb_final': gb_final}

    def save_train_test_data(self):
        """保存训练/测试数据为CSV。"""
        ensure_dir(config.RESULT_DIR)
        for suffix, X_raw, y in [('train', self.X_train, self.y_train),
                                  ('test', self.X_test, self.y_test)]:
            raw = self.scaler.inverse_transform(X_raw)
            df = pd.DataFrame(raw, columns=self.feature_names)
            df['Tc'] = y
            df.to_csv(f"{config.RESULT_DIR}/{suffix}_data.csv", index=False)
