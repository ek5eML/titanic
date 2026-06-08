import os
import random
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

def generate_continuous_feature(
    data: pd.DataFrame,
    base_col_name: str,
    cat_col_name: str,
    bins: list[list[float]],
) -> pd.DataFrame:
    data = data.copy()
    
    for i, (lo, hi) in enumerate(bins):
        data.loc[(data[base_col_name] > lo) & (data[base_col_name] <= hi), cat_col_name] = i

    data.loc[data[base_col_name] <= bins[0][1], cat_col_name] = 0
    data.loc[data[base_col_name] > bins[-1][0], cat_col_name] = i

    data[cat_col_name] = data[cat_col_name].astype('int64')

    return data

def log_run(
  config,
  model_name: str,
  model_params: dict,
  metric_name: str,
  metric_value: float,
  time_s: float,
):
  if not model_params:
    params = {}
  elif isinstance(model_params, DictConfig):
    params = OmegaConf.to_container(model_params, resolve=True) or {}
  else:
    params = dict(model_params)

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
    f'time       : {time_s:.2f} s',
    '',
    'params:',
  ]

  if params:
    max_key_len = max(len(str(key)) for key in params)
    for key, value in sorted(params.items()):
      lines.append(f'  {str(key):<{max_key_len}} : {value}')
  else:
    lines.append('  (default)')

  lines.append('\n')
  block = '\n'.join(lines)

  with log_file.open('a', encoding='utf-8') as file:
    file.write(block)

  print(block)
  return log_file