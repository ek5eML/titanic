import os
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch


def set_seed(seed: int):
  '''
  Set the seed for the random number generators.
  '''
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
  if torch.mps.is_available():
    torch.mps.manual_seed(seed)
  
  os.environ['PYTHONHASHSEED'] = str(seed)

def log_run(
  config,
  model_name: str,
  model_params: dict,
  metric_name: str,
  metric_value: float,
  metric_std: float,
  time_s: float,
):
  params = _normalize_params(model_params)
  timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  log_file = Path('logs') / f'{config.general.experiment_name}.txt'
  log_file.parent.mkdir(parents=True, exist_ok=True)

  lines = [
    '=' * 60,
    timestamp,
    '=' * 60,
    f'experiment : {config.general.experiment_name}',
    f'model      : {model_name}',
    f'metric     : {metric_name} = {metric_value:.6f}',
    f'metric_std : {metric_std:.6f}',
    f'time       : {time_s:.2f} s',
    '',
    *_format_params_block(params),
    '\n',
  ]
  block = '\n'.join(lines)

  with log_file.open('a', encoding='utf-8') as file:
    file.write(block)

  print(block)
  return log_file


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
    rf'metric\s*:\s*{re.escape(metric_name)}\s*=\s*([\d.]+)'
  )
  match = pattern.search(log_file.read_text(encoding='utf-8'))
  if not match:
    return None

  return float(match.group(1))


def save_best_model_params(
  registry_name: str,
  model_params: dict,
  metric_name: str,
  metric_value: float,
  metric_std: float,
):
  '''
  Save params to logs/{registry_name}.txt if metric_value beats the stored score.
  '''
  log_file = Path('logs') / f'{registry_name}.txt'
  log_file.parent.mkdir(parents=True, exist_ok=True)

  stored_score = _read_best_score(log_file, metric_name)
  if (
    stored_score is not None
    and round(metric_value, 6) <= round(stored_score, 6)
  ):    
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