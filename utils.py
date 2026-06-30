import os
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import torch


def set_seed(seed: int):
  """Set random seeds for Python, NumPy, and PyTorch."""
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)

  torch.manual_seed(seed)

  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  if torch.mps.is_available():
    torch.mps.manual_seed(seed)


def get_cv_splitter(config):
  """Return CV depending on the model type."""
  if config.model_type == 'regression':
    return KFold(**config.cv)

  return StratifiedKFold(**config.cv)


def get_scaler(config) -> BaseEstimator | None:
  """Return scaler if config.need_scaler is enabled."""
  if config.need_scaler:
    return StandardScaler()

  return None


def _to_float32_array(X) -> np.ndarray:
  if hasattr(X, 'to_numpy'):
    return X.to_numpy(dtype=np.float32)
  return np.asarray(X, dtype=np.float32)


def fit_scale(scaler, X) -> np.ndarray:
  """Fit scaler on X when present, otherwise return float32 numpy array."""
  X = _to_float32_array(X)
  if scaler is None:
    return X

  return scaler.fit_transform(X)


def transform_scale(scaler, X) -> np.ndarray:
  """Transform X with fitted scaler when present, otherwise return float32 numpy."""
  X = _to_float32_array(X)
  if scaler is None:
    return X

  return scaler.transform(X)


def log_run(
  config,
  model_name: str,
  model_params: dict,
  metric_name: str,
  metric_value: float,
  metric_std: float,
  time_s: float,
  with_header: bool = True,
):
  """Append a formatted run summary to the experiment log and print it."""
  params = _normalize_params(model_params)
  timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  log_file = _log_file(config)

  lines = []
  if with_header:
    lines.extend([
      '=' * 60,
      timestamp,
      '=' * 60,
    ])
  else:
    lines.append('')

  lines.extend([
    f'experiment : {config.general.experiment_name}',
    f'model      : {model_name}',
    f'metric     : {metric_name} = {metric_value:.6f}',
    f'metric_std : {metric_std:.6f}',
    f'time       : {time_s:.2f} s',
    '',
    *_format_params_block(params),
    '\n',
  ])
  block = '\n'.join(lines)

  with log_file.open('a', encoding='utf-8') as file:
    file.write(block)

  print(block)
  return log_file


def _log_file(config) -> Path:
  path = Path(str(config.paths.path_to_logs))
  path.parent.mkdir(parents=True, exist_ok=True)

  return path


def begin_log_section(config, title: str = '') -> Path:
  timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  lines = [
    '=' * 60,
    timestamp,
    '=' * 60,
  ]
  if title:
    lines.append(title)

  block = '\n'.join(lines) + '\n'
  print(block, end='')
  log_file = _log_file(config)
  with log_file.open('a', encoding='utf-8') as file:
    file.write(block)

  return log_file


def log_message(config, message: str) -> None:
  print(message)
  with _log_file(config).open('a', encoding='utf-8') as file:
    file.write(message + '\n')


def _normalize_params(model_params) -> dict:
  if not model_params:
    return {}
  if isinstance(model_params, DictConfig):
    params = OmegaConf.to_container(model_params, resolve=True) or {}
  else:
    params = dict(model_params)

  return params


def _format_params_block(params: dict) -> list[str]:
  lines = ['params:']
  if not params:
    lines.append('  (default)')

    return lines

  max_key_len = max(len(str(key)) for key in params)
  for key, value in sorted(params.items()):
    lines.append(f'  {str(key):<{max_key_len}} : {value}')

  return lines


def _read_best_score(log_file: Path, metric_name: str) -> float | None:
  if not log_file.exists():
    return None

  pattern = re.compile(
    rf'metric\s*:\s*{re.escape(metric_name)}\s*=\s*(-?[\d.]+)'
  )
  match = pattern.search(log_file.read_text(encoding='utf-8'))
  if not match:
    return None

  return float(match.group(1))


def save_best_model_params(
  config,
  registry_name: str,
  model_params: dict,
  metric_name: str,
  metric_value: float,
  metric_std: float,
):
  """Update logs/{model}.txt when CV score improves for the given model."""
  log_file = Path('logs') / f'{registry_name}.txt'
  log_file.parent.mkdir(parents=True, exist_ok=True)

  stored_score = _read_best_score(log_file, metric_name)
  if stored_score is not None:
    if np.isclose(metric_value, stored_score, rtol=0, atol=1e-6):
      return

    is_better = (
      metric_value < stored_score
      if config.is_negative_metric
      else metric_value > stored_score
    )

    if not is_better:
      return

  params = _normalize_params(model_params)
  timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  lines = [
    '=' * 60,
    timestamp,
    '=' * 60,
    f'model      : {registry_name}',
    f'metric     : {metric_name} = {metric_value:.6f}',
    f'metric_std : {metric_std:.6f}',
    '',
    *_format_params_block(params),
    '',
  ]
  block = '\n'.join(lines)

  log_file.write_text(block, encoding='utf-8')
  print(block)
  if stored_score is not None:
    print(f'[{registry_name}] updated: {stored_score:.6f} -> {metric_value:.6f}')
  else:
    print(f'[{registry_name}] saved best score: {metric_value:.6f}')


def save_model(
  model: BaseEstimator,
  path_to_save: str,
):
  """Serialize a fitted estimator to disk with joblib."""
  Path(path_to_save).parent.mkdir(parents=True, exist_ok=True)
  dump(model, path_to_save)


def load_model(path_to_load: str) -> BaseEstimator:
  return load(path_to_load)


def select_best_cv_result(results: list[dict], config) -> dict:
  """Pick the result dict with the best CV metric."""
  best_selector = min if config.is_negative_metric else max

  return best_selector(
    results,
    key=lambda result: result['metric'],
  )


def write_result_md(results: list[dict], config) -> Path:
  """Write a markdown table with CV results for all evaluated models."""
  path = Path(str(config.paths.path_to_results))
  path.parent.mkdir(parents=True, exist_ok=True)

  sorted_results = sorted(
    results,
    key=lambda result: result['metric'],
    reverse=not config.is_negative_metric,
  )

  lines = [
    '| Model | CV | CV STD |',
    '| ----- | -- | ------ |',
  ]
  for result in sorted_results:
    lines.append(
      f"| {result['model_name']} | {result['metric']:.6f} | "
      f"{result['metric_std']:.6f} |"
    )

  path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print(f'Results table written to {path}')

  return path
