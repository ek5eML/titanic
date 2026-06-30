# Titanic

Kaggle [Titanic](https://www.kaggle.com/competitions/titanic) project with a reproducible training pipeline.

## Setup

Recommended: create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Place Kaggle data files in the `data/` directory:

```
data/train.csv
data/test.csv
```

Run the pipeline:

```bash
python main.py
```

On startup, `main.py` quietly installs missing dependencies from `requirements.txt`.

Configure the run in `config.py`:

- `mode` — `train` or `fit`
- `training_model` — model name for `train` mode
- `models_to_evaluate` — list of models for `fit` mode
- `model_type` — `regression` or `classification`
- `need_scaler` — apply `StandardScaler` to features (sklearn and DNN)
- `models_params` — hyperparameters per model

EDA and experiments: `research.ipynb`.

## Workflow

### `train` — cross-validation for one model

- Set `mode: train` and choose a model via `training_model`.
- Runs **StratifiedKFold CV** for classification or **KFold CV** for regression (5 folds by default) on the full training set.
- Writes metrics to `logs/{experiment_name}.txt`.
- If `save_best_model: True`, updates `logs/{model}.txt` when CV score improves.

Checkpoints are not saved in this mode.

### `fit` — full pipeline (CV → best model → submission)

Default mode. One command runs the entire flow:

1. **CV** for every model in `models_to_evaluate`
2. Optional update of `logs/{model}.txt` when `save_best_model: True`
3. Selection of the best model by CV metric
4. Retrain on part of train (`fit.val_size: 0.2` — 20% for validation)
5. `submission.csv` — predictions on test
6. `result.md` — CV summary table (**Model**, **CV**, **CV STD**)

Supported models: `regression`, `ridge`, `lasso`, `elasticnet`, `KNN`, `decision_tree`, `random_forest`, `catboost`, `lightgbm`, `xgboost`, `voting`, `stacking`, `DNN`.

## Results

| Approach              | CV       | CV STD   | LB      | Date       |
| --------------------- | -------- | -------- | ------- | ---------- |
| full catboost         | 0.832729 | 0.024172 | 0.78708 | 2026-06-19 |
| full xgboost          | 0.831611 | 0.019072 | 0.78229 | 2026-06-19 |
| decision tree         | 0.813646 | 0.023955 | 0.78229 | 2026-06-19 |
| DNN FULL DATA         | 0.831611 | 0.022699 | 0.77033 | 2026-06-19 |
| random forest         | 0.821530 | 0.015836 | 0.76794 | 2026-06-19 |
| dnn with batch norm   | 0.834982 | 0.022565 | 0.76315 | 2026-06-19 |
| Ensemble              | 0.831599 | 0.024614 | 0.76315 | 2026-06-18 |
| full lightgbm         | 0.830481 | 0.027137 | 0.76076 | 2026-06-19 |
| KNN                   | 0.825975 | 0.033365 | 0.74401 | 2026-06-17 |
