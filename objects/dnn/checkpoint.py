from pathlib import Path

import torch


def _checkpoint_dir(config) -> Path:
  checkpoint_dir = Path(str(config.paths.path_to_checkpoints))
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  return checkpoint_dir


def get_last_checkpoint_path(config) -> Path:
  return _checkpoint_dir(config) / 'DNN_last.pt'


def get_best_checkpoint_path(config) -> Path:
  return _checkpoint_dir(config) / 'DNN_best.pt'


def save_training_checkpoint(path: Path, state: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(state, path)


def load_training_checkpoint(path: Path) -> dict:
  return torch.load(path, map_location='cpu', weights_only=False)

