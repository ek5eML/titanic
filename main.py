import subprocess
import sys
from pathlib import Path

subprocess.run(
  [sys.executable, '-m', 'pip', 'install', '-r', str(Path(__file__).parent / 'requirements.txt')],
  check=False,
)

import pandas as pd

from config import config
from objects.DataLoader import DataLoader
from objects.Trainer import Trainer
from utils import (
  set_seed,
  save_best_model_params,
)


def fit(config):
  set_seed(config.general.seed)
  trainer = Trainer(config)

  if config.mode == 'fit':
    trainer.fit_full()

  elif config.mode == 'train':
    res = trainer.run_cv()

    if config.save_best_model:
      save_best_model_params(
        registry_name=config.training_model,
        model_params=res['model_params'],
        metric_name=config.metric,
        metric_value=res['metric'],
        metric_std=res['metric_std'],
      )

  elif config.mode == 'submit':
    data_loader = DataLoader(config)
    test_data = data_loader.load_test()

    predictions = trainer.predict(test_data)

    submission = pd.DataFrame({
      config.data.id_col: test_data.index,
      config.data.target_col: predictions.astype(int),
    })
    submission.to_csv(config.paths.path_to_submission, index=False)


if __name__ == '__main__':
  fit(config)
