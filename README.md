# Supercon_Pred — Machine Learning Prediction of Superconducting Transition Temperature (Tc)

A modular Python toolkit for predicting the superconducting transition temperature (Tc) of materials from chemical compositions using machine learning. Built for materials discovery workflows, it supports feature engineering from chemical formulas, ensemble learning, Gaussian process regression, and Bayesian hyperparameter optimization.

## Features

- **Automatic Feature Generation** — Extract 45-dimensional feature vectors from chemical formulas, covering atomic radius, mass, electronegativity, valence electrons, ionization energy, and d-orbital properties, each with 8 statistical descriptors (mean, weighted mean, std, weighted std, range, weighted range, entropy, weighted entropy).
- **Three ML Models** — Random Forest, Gradient Boosting (both with Bayesian hyperparameter search via `skopt`), and Gaussian Process Regression (Rational Quadratic kernel).
- **Comprehensive Evaluation** — R², MAE, MSE, RMSE scores; correlation heatmaps; built-in and permutation feature importance; prediction-vs-true scatter plots.
- **Bayesian Hyperparameter Optimization** — 200-iteration x 10-fold cross-validation search for optimal model parameters.
- **Modular CLI** — Three subcommands (`features`, `train`, `predict`) for a streamlined workflow.
- **Model Persistence** — Trained models and StandardScaler are serialized via pickle for reuse.

## Framework

```
Supercon_Pred/
├── __init__.py              # Package marker
├── config.py                # Central configuration (paths, hyperparameter spaces, parallelism)
├── utils.py                 # Utility functions (directory creation)
├── main.py                  # CLI entry point (argparse)
│
├── data/
│   ├── train/               # Training data directory
│   └── predict/             # Prediction data directory
│
├── features/
│   ├── atoms.py             # Element property dictionary (96 elements) & formula parser
│   └── generator.py         # Feature vector computation (45-dimensional)
│
├── training/
│   ├── models.py            # ModelTrainer: RF / GB / GPR training + BayesSearchCV
│   └── evaluator.py         # Evaluator: scoring metrics, feature importance, plots
│
├── prediction/
│   └── predictor.py         # Predictor: load trained model and predict Tc
│
└── outputs/
    ├── models/              # Serialized model files (.pickle)
    ├── figures/             # Evaluation figures (heatmaps, importance, scatter)
    └── results/             # Scores and prediction CSVs
```

## Implementation Overview

### Feature Engineering

Chemical formulas are parsed via regex, and for each element present, the following properties are retrieved from an internal lookup table (96 elements):

- **Atomic radius** (angstrom)
- **Atomic mass** (amu)
- **Valence electron count**
- **Electronegativity** (Pauling scale)
- **First ionization energy** (eV)
- **Electron shell configuration** (from `chemlib`)

Each property is summarized across all constituent elements with 8 statistical descriptors:

| Descriptor | Unweighted | Stoichiometry-weighted |
|-----------|-----------|----------------------|
| Mean | `mean` | `mean_wtd` |
| Standard deviation | `std` | `std_wtd` |
| Range | `range` | `range_wtd` |
| Entropy | `entropy` | `entropy_wtd` |

Four additional features describe d-orbital properties: shell range (energy level span), d-electron count, d-orbital ratio, and d-orbital unfilled states.

**Total: 1 (number of elements) + 5 x 8 (statistical features) + 4 (d-orbital) = 45 dimensions.**

### Model Training

| Model | Algorithm | Hyperparameter Optimization |
|-------|-----------|---------------------------|
| **Random Forest** | `sklearn.ensemble.RandomForestRegressor` | BayesSearchCV, 200 iterations, 10-fold CV |
| **Gradient Boosting** | `sklearn.ensemble.GradientBoostingRegressor` | BayesSearchCV, 200 iterations, 10-fold CV |
| **Gaussian Process** | `sklearn.gaussian_process.GaussianProcessRegressor` w/ RationalQuadratic kernel | Fixed kernel parameters |

Data is split 90/10 train/test, and features are standardized via `StandardScaler`.

### Evaluation

- R squared, MAE, MSE, RMSE on both train and test sets
- Correlation heatmap of all features
- Built-in feature importance (RF/GB) + permutation importance (all models)
- Top-6 feature vs. Tc scatter plots
- Prediction vs. true value scatter plots

## Installation

### Requirements

- Python 3.10+
- pip

### Install

```bash
git clone https://github.com/your-username/Supercon_Pred.git
cd Supercon_Pred
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
scikit-optimize>=0.9
matplotlib>=3.4
seaborn>=0.11
chemlib>=0.1
```

## Usage

### Workflow Overview

```
[Compound CSV] ---> features ---> [Feature CSV] ---> train ---> [Model + Evaluation]
                                                         |
                                                  predict ---> [Prediction CSV]
```

### 1. Prepare Data

Create a CSV file with at least a `formula` column containing chemical formulas (e.g., `MgB2`, `YBa2Cu3O7`). If you have known Tc values for training, include a `Tc` column.

**Example** (`data/train/materials.csv`):

```csv
formula,Tc
MgB2,39.0
Nb3Sn,18.0
YBa2Cu3O7,93.0
```

**Note**: CSV files are read with GBK encoding (common on Chinese Windows systems). If your file is UTF-8 encoded, the program automatically falls back to UTF-8.

### 2. Generate Features

```bash
python -m Supercon_Pred.main features data/train/materials.csv \
    -o data/train/materials_features.csv \
    --tc-col Tc
```

This produces a 47-column CSV (formula + 45 features + Tc).

### 3. Train Models

```bash
python -m Supercon_Pred.main train data/train/materials_features.csv \
    --models RF,GB,GPR
```

- `--models`: Comma-separated list. Options: `RF`, `GB`, `GPR`. Default: all three.
- Training time depends on dataset size. For ~5000 samples, expect ~30 minutes with Bayesian optimization.
- Trained models are saved to `outputs/models/`.
- Evaluation figures go to `outputs/figures/`.
- Score reports go to `outputs/results/`.

**Optional**: Make predictions immediately after training:

```bash
python -m Supercon_Pred.main train data/train/materials_features.csv \
    --models RF,GB,GPR \
    --predict data/predict/new_materials_features.csv
```

### 4. Predict Tc

```bash
python -m Supercon_Pred.main predict data/predict/new_materials_features.csv \
    --model RandomForest \
    -o results/predictions.csv
```

Model name options: `RandomForest`, `GradientBoosting`, `GaussianProcess`.

### 5. Examine Results

**Score report** (`outputs/results/model_scores.txt`):

```
------ RandomForest -------
train: MAE=2.134, MSE=12.456, RMSE=3.529, R2=0.965
test:  MAE=5.891, MSE=78.234, RMSE=8.845, R2=0.813
----------------
```

**Output figures** (`outputs/figures/`):

| File | Description |
|------|-------------|
| `heatmap.png` | Feature correlation heatmap |
| `feature_importance_*_builtin.png` | Top-10 built-in feature importance |
| `feature_importance_*_permutation.png` | Top-10 permutation importance |
| `scatter_*.png` | Top-6 features vs. Tc scatter plots |
| `*_results_*.png` | Predicted vs. true Tc scatter (train/test) |

**Prediction output** (`outputs/results/prediction_results_*.csv`):

```csv
composition,Tc_prediction
MgB2,38.7
Nb3Sn,17.2
```

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CPU_TOTAL` | `os.cpu_count()` | Total available cores |
| `ModelConfig.n_jobs_search_cv` | `CPU_TOTAL // 4` | Parallel jobs for hyperparameter search |
| `RF_PARAM_SPACE` | `n_estimators: [20,200]`, etc. | Random Forest search space |
| `GB_PARAM_SPACE` | `n_estimators: [20,500]`, etc. | Gradient Boosting search space |

## Project Structure

- `Supercon_Pred/` — Main Python package
- `data/train/` — Training data
- `data/predict/` — Data for prediction
- `outputs/` — Generated outputs (models, figures, results)

## Citation

If you use this program in your research, please cite:

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

## License

This project is provided for academic research purposes.
