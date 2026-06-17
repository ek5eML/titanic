import time
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from objects.DataLoader import DataLoader
from objects.pipeline_builder import build_pipeline
from utils import begin_log_section, log_run, load_model, save_model


class SklearnRunner:
  def __init__(self, config):
    self.config = config

  def run_cv(self, name: str = '', params: dict | None = None) -> dict:
    if self.config.logging:
      begin_log_section(self.config)

    data_loader = DataLoader(self.config)
    train_data = data_loader.load_train()
    X, y = data_loader.split_data(train_data)

    pipeline = build_pipeline(self.config, name, params)
    cv = StratifiedKFold(**self.config.cv)

    start = time.perf_counter()
    res = cross_validate(
      pipeline,
      X, y,
      cv=cv,
      scoring=self.config.metric,
    )
    elapsed_s = time.perf_counter() - start
    model = pipeline.named_steps['model']

    result = {
      'model_name': model.__class__.__name__,
      'model_params': model.get_params(),
      'metric': res['test_score'].mean(),
      'metric_std': res['test_score'].std(),
      'fold_metrics': res['test_score'].tolist(),
      'time_s': elapsed_s,
    }

    if self.config.logging:
      log_run(
        config=self.config,
        model_name=result['model_name'],
        model_params=result['model_params'],
        metric_name=self.config.metric,
        metric_value=result['metric'],
        metric_std=result['metric_std'],
        time_s=result['time_s'],
        with_header=False,
      )

    return result

  def fit_full(self, name: str = '', params: dict | None = None):
    start = time.perf_counter()

    if self.config.logging:
      begin_log_section(self.config)

    data_loader = DataLoader(self.config)
    train_data = data_loader.load_train()
    X, y = data_loader.split_data(train_data)

    X_train, X_val, y_train, y_val = train_test_split(
      X,
      y,
      test_size=self.config.fit.val_size,
      stratify=y,
      random_state=self.config.general.seed,
    )

    pipeline = build_pipeline(self.config, name, params)
    pipeline.fit(X_train, y_train)
    val_metric = pipeline.score(X_val, y_val)

    elapsed_s = time.perf_counter() - start
    model = pipeline.named_steps['model']

    if self.config.logging:
      log_run(
        config=self.config,
        model_name=model.__class__.__name__,
        model_params=model.get_params(),
        metric_name=self.config.metric,
        metric_value=val_metric,
        metric_std=0.0,
        time_s=elapsed_s,
        with_header=False,
      )

    path_to_model = (
      Path(str(self.config.paths.path_to_checkpoints))
      / f'{self.config.training_model}.joblib'
    )
    save_model(pipeline, str(path_to_model))

    return pipeline

  def predict(self, test_data):
    force_refit_models = {'xgboost', 'lightgbm'}

    if self.config.rerun or self.config.training_model in force_refit_models:
      pipeline = self.fit_full()
      return pipeline.predict(test_data)

    path_to_model = (
      Path(str(self.config.paths.path_to_checkpoints))
      / f'{self.config.training_model}.joblib'
    )
    pipeline = load_model(str(path_to_model))
    return pipeline.predict(test_data)
