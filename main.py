import subprocess
import sys
from pathlib import Path


def install_requirements() -> None:
  """Install project dependencies from requirements.txt via pip."""
  path = Path(__file__).resolve().parent / 'requirements.txt'
  subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-qr', str(path)],
    stdout=subprocess.DEVNULL,
  )


def run_train(config):
  """Run cross-validation for a single model from config.training_model."""
  from objects.Trainer import Trainer
  from utils import save_best_model_params

  trainer = Trainer(config)
  res = trainer.run_cv()

  if config.save_best_model:
    save_best_model_params(
      config=config,
      registry_name=config.training_model,
      model_params=res['model_params'],
      metric_name=config.metric,
      metric_value=res['metric'],
      metric_std=res['metric_std'],
    )


def main(config):
  """Entry point: dispatch to train or fit mode."""
  from objects.fit_pipeline import run_fit
  from utils import set_seed

  set_seed(config.general.seed)

  if config.mode == 'train':
    run_train(config)
  elif config.mode == 'fit':
    run_fit(config)
  else:
    raise ValueError(
      f"Unknown mode: {config.mode!r}. Supported modes: 'train', 'fit'."
    )


if __name__ == '__main__':
  install_requirements()
  from config import config

  main(config)
