from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from objects.DataLoader import DataLoader
from objects.Trainer import Trainer
from utils import (
  save_best_model_params,
  select_best_cv_result,
  write_result_md,
)


def get_models_to_evaluate(config) -> list[str]:
  """Return model names to compare in fit mode."""
  if config.get('models_to_evaluate'):
    return list(config.models_to_evaluate)

  return list(config.models_params.keys())


def run_cv_for_all_models(config) -> list[dict]:
  """Run CV for every model in models_to_evaluate and optionally save best params."""
  models = get_models_to_evaluate(config)
  cv_results: list[dict] = []

  for model_name in models:
    OmegaConf.update(config, 'training_model', model_name)
    trainer = Trainer(config)
    res = trainer.run_cv()

    result = {
      'model_name': model_name,
      'metric': res['metric'],
      'metric_std': res['metric_std'],
      'model_params': res['model_params'],
    }
    cv_results.append(result)

    if config.save_best_model:
      save_best_model_params(
        config=config,
        registry_name=model_name,
        model_params=res['model_params'],
        metric_name=config.metric,
        metric_value=res['metric'],
        metric_std=res['metric_std'],
      )

  return cv_results


def train_best_model_and_predict(config, best_model_name: str) -> pd.Series:
  """Retrain the best model on train/val split and predict on the test set."""
  OmegaConf.update(config, 'training_model', best_model_name)
  trainer = Trainer(config)

  data_loader = DataLoader(config)
  test_data = data_loader.load_test()

  if best_model_name == 'DNN':
    trainer.fit_full()
    predictions = trainer.predict(test_data)
  else:
    pipeline = trainer.fit_full()
    predictions = pipeline.predict(test_data)

  return pd.Series(predictions, index=test_data.index, name=config.data.target_col)


def save_submission(config, predictions: pd.Series) -> Path:
  """Write predictions to submission.csv."""
  submission = pd.DataFrame({
    config.data.id_col: predictions.index,
    config.data.target_col: predictions.astype(int),
  })
  path = Path(str(config.paths.path_to_submission))
  submission.to_csv(path, index=False)

  return path


def run_fit(config) -> dict:
  """Full fit pipeline: CV all models, retrain best, save submission and result.md."""
  cv_results = run_cv_for_all_models(config)
  best = select_best_cv_result(cv_results, config)

  print(
    f"Best model: {best['model_name']} | "
    f"{config.metric}={best['metric']:.6f} (+/- {best['metric_std']:.6f})"
  )

  predictions = train_best_model_and_predict(config, best['model_name'])
  submission_path = save_submission(config, predictions)
  results_path = write_result_md(cv_results, config)

  print(f'Submission saved to {submission_path}')
  print(f'Results saved to {results_path}')

  return {
    'cv_results': cv_results,
    'best_model': best,
    'submission_path': submission_path,
    'results_path': results_path,
  }
