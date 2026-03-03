# Fault Detection ML Project

This repository contains a complete, production-ready machine learning pipeline for **binary fault detection** using tabular sensor data.

The goal is to classify each sample as:

- `0` = Normal
- `1` = Faulty

The project is designed to be **modular**, **reproducible**, and easy to extend for additional models or preprocessing steps.

## Dataset

- **Training file**: `data/TRAIN.csv`
- **Test file**: `data/TEST.csv`
- **Features**: 47 numerical features named `F01` to `F47`
- **Target column**: `Class` (0/1)
- **Test file ID column**: `ID` (used for submission, not as a feature)

The final competition-style submission must have the format:

```text
ID,CLASS
<id_1>,<prediction_1>
<id_2>,<prediction_2>
...
```

## Project Structure

```text
fault-detection-ml/
│
├── data/
│   ├── TRAIN.csv
│   └── TEST.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   ├── final_model.pkl
│   └── feature_importance.png
│
├── submissions/
│   └── final_submission.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Models

The following models are implemented:

- **Logistic Regression** (baseline)
- **Random Forest**
- **XGBoost**
- **LightGBM**

All models:

- Accept configurable hyperparameters
- Support basic class imbalance handling
- Are wrapped in a scikit-learn compatible API

## Cross-Validation and Model Selection

Training uses **Stratified 5-Fold Cross-Validation** to preserve class proportions in each fold.

For each model, the following metrics are computed:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**

Each metric is reported as **mean ± standard deviation** across folds.

The **primary selection criterion** is **ROC-AUC**. The model with the best mean ROC-AUC is considered the best base model.

### Hyperparameter Tuning

For **XGBoost** and **LightGBM**, we perform **RandomizedSearchCV** (20–30 iterations) with stratified cross-validation and ROC-AUC as the scoring metric.

Tuned hyperparameters:

- **XGBoost**
  - `n_estimators`
  - `max_depth`
  - `learning_rate`
  - `subsample`
  - `colsample_bytree`

- **LightGBM**
  - `num_leaves`
  - `learning_rate`
  - `n_estimators`
  - `feature_fraction`
  - `bagging_fraction`

The script reports the **best parameters** and **best CV ROC-AUC score** for each boosting model.

## Feature Importance

For the best boosting model (XGBoost or LightGBM, based on tuned ROC-AUC), the pipeline:

- Trains the model on the full training set
- Extracts feature importances
- Saves a bar plot to `models/feature_importance.png`

## Final Model and Prediction Pipeline

The **final model** (including preprocessing and estimator) is trained on the **full training dataset** and saved to:

- `models/final_model.pkl`

The prediction pipeline in `src/predict.py`:

1. Loads `models/final_model.pkl`
2. Loads `data/TEST.csv`
3. Applies the **same preprocessing** used in training (via the saved pipeline)
4. Generates predictions
5. Saves results to:
   - `submissions/final_submission.csv`

Output format:

```text
ID,CLASS
<id_1>,<prediction_1>
<id_2>,<prediction_2>
...
```

The order of IDs in the submission matches the order in `TEST.csv`.

## EDA Notebook

The notebook `notebooks/eda.ipynb` performs:

- Basic data inspection:
  - Shape
  - `head()`
  - `info()`
  - Null values
- Class distribution bar plot
- Correlation heatmap
- Feature distributions
- Identification of:
  - Constant columns
  - Highly correlated feature pairs
  - Potential outliers

Key visualizations are saved within the notebook directory.

## Reproducibility and Code Quality

- **No hardcoded absolute paths**: all paths are derived from `src/config.py` using `pathlib`.
- **Random seeds** are set for:
  - Python `random`
  - NumPy
  - scikit-learn
  - XGBoost / LightGBM where applicable
- **Logging** (instead of print) is used in pipeline scripts (`train.py`, `predict.py`, etc.).
- **No data leakage**:
  - Preprocessing (imputation, scaling, feature selection) is done inside scikit-learn `Pipeline`s that are cross-validated.
  - The same fitted preprocessing is serialized with the model.

## Setup Instructions

1. **Clone or copy the project**, then ensure the data files are present:

   - Place `TRAIN.csv` and `TEST.csv` into the `data/` directory.

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run training** (cross-validation, tuning, final model training, and feature importance):

   ```bash
   python src/train.py
   ```

5. **Generate predictions for the test set**:

   ```bash
   python src/predict.py
   ```

   This will create `submissions/final_submission.csv`.

## Model Comparison (Example Template)

Once you have run `train.py`, you can fill in a comparison table like the one below in this README (or another report):

| Model            | Accuracy (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) | ROC-AUC (mean ± std) |
|------------------|-----------------------|-------------------------|----------------------|------------------|-----------------------|
| LogisticRegression | ...                 | ...                     | ...                  | ...              | ...                   |
| RandomForest     | ...                   | ...                     | ...                  | ...              | ...                   |
| XGBoost          | ...                   | ...                     | ...                  | ...              | ...                   |
| LightGBM         | ...                   | ...                     | ...                  | ...              | ...                   |

The **final model** used in production/inference is whichever boosting model achieves the best tuned ROC-AUC.

