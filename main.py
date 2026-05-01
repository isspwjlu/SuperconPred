"""Supercon_Pred CLI - Superconducting transition temperature (Tc) prediction tool.

Usage:
    # Generate features from compound list
    python main.py features <input_csv> [-o output.csv] [--tc-col Tc]

    # Train models (RF / GB / GPR)
    python main.py train <feature_csv> [--models RF,GB,GPR] [--predict data.csv]

    # Predict Tc using trained model
    python main.py predict <feature_csv> --model RandomForest [-o result.csv]
"""
import argparse
import pandas as pd
import config
from utils import ensure_dir
from features.generator import FeatureGenerator
from training.models import ModelTrainer
from training.evaluator import Evaluator
from prediction.predictor import Predictor


def cmd_features(args):
    """子命令: features - 从化合物列表生成特征。"""
    gen = FeatureGenerator()
    gen.generate(args.input_csv, args.output or args.input_csv.replace('.csv', '_features.csv'),
                 tc_col=args.tc_col)


def cmd_train(args):
    """子命令: train - 训练模型。"""
    models = args.models.split(',') if args.models else ['RF', 'GB', 'GPR']
    trainer = ModelTrainer(args.feature_csv,
                           enable_rf='RF' in models,
                           enable_gb='GB' in models,
                           enable_gpr='GPR' in models)
    final = trainer.train_all()
    trainer.save_train_test_data()

    df = pd.read_csv(args.feature_csv, header=0, encoding='gbk').dropna()
    if df.dtypes[df.keys()[0]] == 'object':
        df.drop(df.columns[0], axis=1, inplace=True)
    feature_names = trainer.feature_names

    ev = Evaluator(df, feature_names)
    ev.plot_correlation_heatmap(df, f"{config.FIGURE_DIR}/heatmap.png")

    if 'RF' in models and final.get('rf_final'):
        ev.plot_feature_importance(final['rf_final'], trainer.X_train, trainer.y_train, 'RandomForest')
        ev.plot_prediction_scatter(trainer.y_train, trainer.y_train_pred['RandomForest'], 'RandomForest', 'train')
        ev.plot_prediction_scatter(trainer.y_test, trainer.y_test_pred['RandomForest'], 'RandomForest', 'test')

    if 'GB' in models and final.get('gb_final'):
        ev.plot_feature_importance(final['gb_final'], trainer.X_train, trainer.y_train, 'GradientBoosting')
        ev.plot_prediction_scatter(trainer.y_train, trainer.y_train_pred['GradientBoosting'], 'GradientBoosting', 'train')
        ev.plot_prediction_scatter(trainer.y_test, trainer.y_test_pred['GradientBoosting'], 'GradientBoosting', 'test')

    if 'GPR' in models and 'GaussianProcess' in trainer.models:
        ev.plot_prediction_scatter(trainer.y_train, trainer.y_train_pred['GaussianProcess'], 'GaussianProcess', 'train')
        ev.plot_prediction_scatter(trainer.y_test, trainer.y_test_pred['GaussianProcess'], 'GaussianProcess', 'test')
        ev.plot_feature_importance(trainer.models['GaussianProcess'], trainer.X_train, trainer.y_train, 'GaussianProcess')

    ev.save_scores(trainer.y_train_pred, trainer.y_test_pred,
                   trainer.y_train, trainer.y_test,
                   f"{config.RESULT_DIR}/model_scores.txt")
    print(f"[完成] 训练完成，所有输出保存至 {config.OUTPUT_DIR}")

    if args.predict:
        predictor = Predictor()
        for name in trainer.models:
            predictor.predict(args.predict, name, scaler=trainer.scaler)


def cmd_predict(args):
    """子命令: predict - 使用已训练模型预测 Tc。"""
    Predictor().predict(args.feature_csv, args.model, output_file=args.output)


def main():
    ensure_dir(config.OUTPUT_DIR)
    for d in [config.MODEL_DIR, config.FIGURE_DIR, config.RESULT_DIR]:
        ensure_dir(d)

    parser = argparse.ArgumentParser(description='Supercon_Pred - 超导材料 Tc 预测工具')
    sub = parser.add_subparsers(dest='command')

    pf = sub.add_parser('features', help='从化合物列表生成特征')
    pf.add_argument('input_csv', help='输入CSV（需包含 formula 列）')
    pf.add_argument('-o', '--output', help='输出特征CSV路径')
    pf.add_argument('--tc-col', default='Tc', help='Tc列名')

    pt = sub.add_parser('train', help='训练模型')
    pt.add_argument('feature_csv', help='特征数据CSV')
    pt.add_argument('--models', default=None, help='逗号分隔: RF,GB,GPR')
    pt.add_argument('--predict', default=None, help='训练后预测的特征CSV')

    pp = sub.add_parser('predict', help='预测 Tc')
    pp.add_argument('feature_csv', help='预测用特征数据CSV')
    pp.add_argument('--model', required=True, help='模型名称')
    pp.add_argument('-o', '--output', help='输出结果CSV')

    args = parser.parse_args()
    handler = {'features': cmd_features, 'train': cmd_train, 'predict': cmd_predict}.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
