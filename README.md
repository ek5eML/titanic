# House Prices

## Setup

Create a virtual environment and install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

Configure the model and run mode in `config.py` (`training_model`, `mode`).

## Workflow

Set the run mode in `config.py` via `mode`. Choose the model via `training_model`.

### `train` — cross-validation

- Loads the training set and runs **StratifiedKFold CV** (5 folds by default) on **all** training data.
- Computes the mean metric and std across folds, and writes the result to `logs/{experiment_name}.txt`.
- If `save_best_model: True`, updates `logs/{model}.txt` with the best parameters (when the CV score improves).

The model is **not** saved to checkpoints in this mode.

### `fit` — training and saving

- Loads the training set and splits it into **train/val** (`val_size: 0.2`, i.e. 80% / 20%) — **not on the full dataset**.
- Trains the pipeline on the train split, evaluates the metric on val, and logs to `logs/{experiment_name}.txt`.
- Saves the trained pipeline to `checkpoints/{training_model}.joblib`.

### `submit` — test prediction

- Loads the test set.
- Loads the saved model from `checkpoints/{training_model}.joblib` and runs `predict`.
- Writes the output to `submission.csv`.

Exception: for `xgboost`, `lightgbm`, `ensemble`, or when `rerun: True`, the model is **retrained** via `fit` before prediction (instead of loading from checkpoint).

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



