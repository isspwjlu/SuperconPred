"""Tc 预测器：加载已训练模型预测新材料的超导转变温度。"""
import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config
from utils import ensure_dir


class Predictor:
    """Tc 预测器。"""

    def __init__(self, model_dir: str = config.MODEL_DIR,
                 result_dir: str = config.RESULT_DIR):
        self.model_dir = model_dir
        self.result_dir = result_dir

    def predict(self, feature_file: str, model_name: str,
                scaler: StandardScaler | None = None,
                output_file: str | None = None) -> str:
        """使用已训练模型对特征数据进行 Tc 预测。

        Args:
            feature_file: 特征数据 CSV 路径
            model_name: 模型名称 ('RandomForest', 'GradientBoosting', 'GaussianProcess')
            scaler: 训练时的 StandardScaler（为None时从训练数据学习）
            output_file: 输出结果路径

        Returns:
            输出结果路径
        """
        if not re.match(r'^[A-Za-z0-9_]+$', model_name):
            raise ValueError(f"非法模型名称: {model_name}")

        scaler_path = os.path.join(self.model_dir, 'scaler.pickle')
        model_path = os.path.join(self.model_dir, f'{model_name}_model.pickle')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在，请先训练模型: {model_path}")

        if scaler is None:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        try:
            df = pd.read_csv(feature_file, header=0, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(feature_file, header=0, encoding='utf-8')
        if df.dtypes[df.keys()[0]] == 'object':
            formulas = df[df.columns[0]].values
            df.drop(df.columns[0], axis=1, inplace=True)
        else:
            formulas = np.arange(len(df))

        y_pred = model.predict(scaler.transform(df.values.astype(float)))

        output_file = output_file or f"{self.result_dir}/prediction_results_{model_name}.csv"
        ensure_dir(self.result_dir)
        pd.DataFrame({'composition': formulas, 'Tc_prediction': y_pred}).to_csv(output_file, index=False)
        print(f"[预测] 结果已保存: {output_file}")
        return output_file
