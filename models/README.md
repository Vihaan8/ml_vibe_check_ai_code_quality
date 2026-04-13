# Models

This directory contains all model training code and saved outputs for the Vibe Check defect prediction task. The goal is binary classification: given a code sample and its static features, predict whether it passes its test suite.

Both model versions are trained from the project root using `main.py` or by running the scripts directly.


## Version 1: Static Features Only

**Script**: `train_models_v1.py`

Trains Logistic Regression and LightGBM on the 17 hand-crafted static features from feature extraction. This is the baseline experiment to see how far classical software metrics, AST structure, prompt alignment, and LLM smell features can take us.

**Logistic Regression** is tuned over regularization strength C in {0.001, 0.01, 0.1, 1, 10, 100} with balanced class weights and an LBFGS solver. The model is wrapped in a StandardScaler pipeline.

**LightGBM** is tuned over n_estimators {200, 500}, learning_rate {0.05, 0.1}, and max_depth {4, 7} with early stopping on the validation set. Class imbalance is handled via scale_pos_weight.

Results on the held-out test set:

| Model | AUC-ROC | F1 |
|---|---|---|
| Logistic Regression | 0.604 | 0.538 |
| LightGBM | 0.608 | 0.516 |

Outputs saved to `outputs_v1/`:
- `logreg_model.pkl`, `lgbm_model.pkl` (trained models)
- `metrics.txt` (AUC, F1, classification reports)
- `results.csv` (test set predictions and probabilities)
- `logreg_coefs.png` (feature weight chart)
- `lgbm_shap.png` (SHAP feature importance)
- `pr_curves.png` (precision-recall curves)


## Version 2: Static Features + TF-IDF

**Script**: `train_models_v2.py`

Extends v1 by adding TF-IDF features extracted directly from the raw generated code. This gives models access to actual code tokens (function names, keywords, syntax patterns) rather than just summary statistics.

Two TF-IDF vectorizers are fit on the training set only:
- Word-level (1-2 grams, 10,000 features): captures identifiers and keywords
- Character-level (2-4 grams, 10,000 features): captures syntax patterns

Combined with the 17 static features, this gives 20,017 total features.

**Logistic Regression** is tuned over C in {0.01, 0.1, 1, 10} with a SAGA solver for efficiency on the larger feature matrix.

**LightGBM** is tuned over n_estimators {300, 500} and learning_rate {0.05, 0.1} with colsample_bytree=0.3 (lower than v1 because most features are TF-IDF).

**Random Forest** uses only the 17 static features (RF on 20K sparse TF-IDF columns is prohibitively slow). Tuned over n_estimators {200, 500} and max_depth {8, 15, None}.

Results on the held-out test set:

| Model | AUC-ROC | F1 |
|---|---|---|
| Logistic Regression | 0.635 | 0.537 |
| LightGBM | 0.634 | 0.529 |
| Random Forest | 0.605 | 0.537 |

Adding TF-IDF improved AUC by about 0.03 for Logistic Regression and LightGBM. Random Forest, which only uses static features, performs comparably to v1.

Outputs saved to `outputs_v2/`:
- `logreg_model.pkl`, `lgbm_model.pkl`, `rf_model.pkl` (trained models)
- `word_tfidf.pkl`, `char_tfidf.pkl` (fitted TF-IDF vectorizers)
- `metrics.txt` (AUC, F1, classification reports)
- `results.csv` (test set predictions and probabilities)
- `feature_importance.png` (top TF-IDF tokens by logistic regression coefficient)
- `pr_curves.png` (precision-recall curves for all three models)


## Class Imbalance

The dataset is roughly 41% pass / 59% fail. All models handle this through class weighting:
- Logistic Regression: `class_weight="balanced"`
- LightGBM: `scale_pos_weight` set to the negative/positive ratio
- Random Forest: `class_weight="balanced"`

Evaluation uses AUC-ROC and F1 rather than accuracy, since accuracy is misleading when classes are imbalanced.


## Feature Importance

The most predictive static features are:

| Feature | Correlation with label | Direction |
|---|---|---|
| classical_loc | -0.174 | Longer code fails more |
| ast_has_error_handling | -0.122 | More try/except correlates with failure |
| classical_cyclomatic_complexity | -0.118 | More complex code fails more |
| ast_try_count | -0.116 | Same as above |
| classical_max_nesting_depth | -0.104 | Deeper nesting fails more |

The negative correlations mean that as solutions grow longer and more complex, they are more likely to fail. This suggests LLMs struggle with task complexity: simple tasks get correct short answers, while harder tasks produce longer but incorrect code.

Prompt-code alignment features (library coverage, missing libs) show near-zero correlation, indicating that simple heuristic alignment is not enough to capture whether generated code satisfies the prompt.


## How to Run

```bash
# From the project root
python main.py --models v1    # static features only
python main.py --models v2    # static + TF-IDF
python main.py --models       # both

# Or run scripts directly
python models/train_models_v1.py
python models/train_models_v2.py
```

Both scripts expect the split CSVs to exist at `data/clean/splits/`. Run `python main.py --preprocess --features` first if they do not.
