# Phase 2: Defect Prediction Models

This folder contains the Phase 2 modeling pipeline for the Vibe Check project.

The goal of Phase 2 is to train models that predict whether LLM-generated code will pass its test suite, using the static features extracted in Phase 1.

## Files

**train_models.py**
Version 1. Trains Logistic Regression and LightGBM on the 17 static features from Phase 1 alone.

**train_models_v2.py**
Version 2 (best results). Trains Logistic Regression, LightGBM, and Random Forest on a combination of the 17 static features plus TF-IDF features extracted directly from the raw code text. TF-IDF captures code patterns like function names, keywords, and syntax that the static features miss.

## Models

### Version 1 — Static features only
Input: 17 static features from Phase 1 (complexity, AST counts, alignment, LLM smells)

| Model | AUC-ROC | F1 |
|---|---|---|
| Logistic Regression | 0.6042 | 0.5384 |
| LightGBM | 0.6076 | 0.5160 |

### Version 2 — Static features + TF-IDF (recommended)
Input: 17 static features + 10,000 word n-gram TF-IDF features + 10,000 character n-gram TF-IDF features = 20,017 total features

| Model | AUC-ROC | F1 |
|---|---|---|
| Logistic Regression | 0.6354 | 0.5370 |
| LightGBM | 0.6336 | 0.5291 |
| Random Forest | 0.6051 | 0.5374 |

Adding TF-IDF improved AUC by ~0.03 across all models by giving them access to the actual code text rather than just summary statistics.

## Feature design

**Static features (17)** come directly from Phase 1 feature extraction. The features most correlated with pass/fail were:
- `classical_loc` — longer code tends to pass more often
- `ast_has_error_handling` — code with try/except tends to pass more often
- `classical_cyclomatic_complexity` — more complex code tends to pass more often

**TF-IDF features** are computed from the raw generated code using two vectorizers:
- Word-level (1–2 grams, 10,000 features): captures identifiers and function names
- Character-level (2–4 grams, 10,000 features): captures syntax patterns and operators

Both vectorizers are fit on the training set only and applied to val and test to prevent data leakage.

## Handling class imbalance

The dataset has roughly 41% pass / 59% fail. Both models are trained with class weighting to prevent them from simply predicting the majority class:
- Logistic Regression: `class_weight="balanced"`
- LightGBM: `scale_pos_weight` set to the negative/positive ratio (~1.48)
- Random Forest: `class_weight="balanced"`

Models are evaluated with AUC-ROC and F1 rather than raw accuracy, since accuracy is misleading on imbalanced datasets.

## Hyperparameter tuning

Both versions tune hyperparameters using the validation set only. The test set is never used during training or tuning — only for final evaluation.

Version 1 tunes:
- Logistic Regression: regularization strength C ∈ {0.001, 0.01, 0.1, 1, 10, 100}
- LightGBM: n_estimators ∈ {200, 500}, learning_rate ∈ {0.05, 0.1}, max_depth ∈ {4, 7}

Version 2 tunes:
- Logistic Regression: C ∈ {0.01, 0.1, 1, 10}
- LightGBM: n_estimators ∈ {300, 500}, learning_rate ∈ {0.05, 0.1}
- Random Forest: n_estimators ∈ {200, 500}, max_depth ∈ {8, 15, None}

## Output

Each model version saves its outputs into its own folder.

`models/` (v1 outputs):
- `logreg_model.pkl` — trained Logistic Regression
- `lgbm_model.pkl` — trained LightGBM
- `results.csv` — test-set predictions and probabilities for every sample
- `metrics.txt` — AUC, F1, and full classification report
- `logreg_coefs.png` — feature weight chart
- `lgbm_shap.png` — SHAP feature importance chart
- `pr_curves.png` — precision-recall curve

`models_v2/` (v2 outputs):
- `logreg_model.pkl` — trained Logistic Regression
- `lgbm_model.pkl` — trained LightGBM
- `rf_model.pkl` — trained Random Forest
- `word_tfidf.pkl` — fitted word TF-IDF vectorizer
- `char_tfidf.pkl` — fitted character TF-IDF vectorizer
- `results.csv` — test-set predictions and probabilities for every sample
- `metrics.txt` — AUC, F1, and full classification report
- `feature_importance.png` — top TF-IDF tokens by logistic regression coefficient
- `pr_curves.png` — precision-recall curve for all three models

## Installation

```
pip install pandas numpy scikit-learn lightgbm shap matplotlib scipy
```

## Usage

Run from the project root. Phase 1 feature extraction must be completed first (see `data/phase1/README.md`).

```
# Version 1 — static features only
python3 data/phase2/train_models.py

# Version 2 — static + TF-IDF (recommended)
python3 data/phase2/train_models_v2.py
```

Both scripts expect the following files to exist:
```
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
data/splits/train_features.csv
data/splits/val_features.csv
data/splits/test_features.csv
```
