from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from objects.dnn.checkpoint import (
  load_training_checkpoint,
  save_training_checkpoint,
)
from utils import log_message


def _get_device(device_name: str) -> torch.device:
  if device_name != 'auto':
    return torch.device(device_name)
  if torch.cuda.is_available():
    return torch.device('cuda')
  if torch.backends.mps.is_available():
    return torch.device('mps')
  return torch.device('cpu')


def _get_preds(logits: torch.Tensor) -> torch.Tensor:
  if logits.ndim == 1:
    return (torch.sigmoid(logits) >= 0.5).long()

  return logits.argmax(dim=-1)


def train_epoch(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
) -> float:
  model.train()
  total_loss = 0.0

  for batch_x, batch_y in dataloader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * len(batch_x)

  return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  device: torch.device,
) -> tuple[float, float]:
  model.eval()
  total_loss = 0.0
  correct = 0

  for batch_x, batch_y in dataloader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    total_loss += loss.item() * len(batch_x)
    preds = _get_preds(logits)
    correct += (preds == batch_y).sum().item()

  dataset_size = len(dataloader.dataset)
  avg_loss = total_loss / dataset_size
  accuracy = correct / dataset_size

  return avg_loss, accuracy


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
  return optimizer.param_groups[0]['lr']


def _build_checkpoint_state(
  *,
  epoch: int,
  model: nn.Module,
  optimizer: torch.optim.Optimizer,
  scheduler: torch.optim.lr_scheduler.LRScheduler | None,
  best_val_accuracy: float,
  best_val_loss: float,
  best_model_state: dict | None,
  last_val_loss: float,
  epochs_without_improvement: int,
  history: dict,
  scaler=None,
  model_state_dict: dict | None = None,
) -> dict:
  return {
    'epoch': epoch,
    'model_state_dict': (
      model_state_dict
      if model_state_dict is not None
      else model.state_dict()
    ),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': (
      scheduler.state_dict() if scheduler is not None else None
    ),
    'best_val_accuracy': best_val_accuracy,
    'best_val_loss': best_val_loss,
    'best_model_state_dict': best_model_state,
    'last_val_loss': last_val_loss,
    'epochs_without_improvement': epochs_without_improvement,
    'history': history,
    'scaler': scaler,
  }


def _restore_training_state(
  checkpoint: dict,
  model: nn.Module,
  optimizer: torch.optim.Optimizer,
  scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> dict:
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  if (
    scheduler is not None
    and checkpoint.get('scheduler_state_dict') is not None
  ):
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

  return {
    'start_epoch': int(checkpoint['epoch']) + 1,
    'best_val_accuracy': float(checkpoint['best_val_accuracy']),
    'best_val_loss': float(checkpoint.get('best_val_loss', float('inf'))),
    'best_model_state': checkpoint.get('best_model_state_dict'),
    'last_val_loss': float(checkpoint['last_val_loss']),
    'epochs_without_improvement': int(checkpoint['epochs_without_improvement']),
    'history': checkpoint.get('history', {
      'train_loss': [],
      'val_loss': [],
      'val_accuracy': [],
      'lr': [],
    }),
    'scaler': checkpoint.get('scaler'),
  }


def fit_model(
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: DataLoader,
  *,
  lr: float,
  weight_decay: float,
  num_epochs: int,
  patience: int,
  device_name: str,
  config=None,
  fold: int | None = None,
  scaler=None,
  last_checkpoint_path: Path | None = None,
  best_checkpoint_path: Path | None = None,
  resume: bool = False,
) -> tuple[float, float]:
  device = _get_device(device_name)
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
  )

  best_accuracy = 0.0
  best_val_loss = float('inf')
  best_state = None
  epochs_without_improvement = 0
  last_val_loss = 0.0
  start_epoch = 1
  history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'lr': [],
  }

  should_log = config is not None and config.logging
  should_save_checkpoints = (
    config is not None
    and config.save_last_model
    and last_checkpoint_path is not None
    and best_checkpoint_path is not None
  )
  fold_prefix = f'fold {fold} | ' if fold is not None else ''

  if resume and last_checkpoint_path is not None and last_checkpoint_path.exists():
    checkpoint = load_training_checkpoint(last_checkpoint_path)
    restored = _restore_training_state(checkpoint, model, optimizer, scheduler)
    start_epoch = restored['start_epoch']
    best_accuracy = restored['best_val_accuracy']
    best_val_loss = restored['best_val_loss']
    best_state = restored['best_model_state']
    last_val_loss = restored['last_val_loss']
    epochs_without_improvement = restored['epochs_without_improvement']
    history = restored['history']
    history.setdefault('lr', [])
    if scaler is None:
      scaler = restored['scaler']

    if should_log:
      log_message(
        config,
        (
          f'{fold_prefix}resumed from epoch {start_epoch} | '
          f'best_val_acc={best_accuracy:.6f}'
        ),
      )
  elif should_log:
    log_message(
      config,
      f'{fold_prefix}training started | epochs={num_epochs} patience={patience}',
    )

  for epoch in range(start_epoch, num_epochs + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    last_val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(last_val_loss)
    history['val_accuracy'].append(val_accuracy)

    scheduler.step(last_val_loss)
    current_lr = _get_lr(optimizer)
    history['lr'].append(current_lr)

    if should_log:
      log_message(
        config,
        (
          f'{fold_prefix}epoch {epoch:3d} | '
          f'train_loss={train_loss:.6f} | '
          f'val_loss={last_val_loss:.6f} | '
          f'val_acc={val_accuracy:.6f} | '
          f'lr={current_lr:.6f}'
        ),
      )

    if val_accuracy > best_accuracy:
      best_accuracy = val_accuracy
      best_val_loss = last_val_loss
      best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
      }
      epochs_without_improvement = 0

      if should_save_checkpoints:
        checkpoint_state = _build_checkpoint_state(
          epoch=epoch,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          best_val_accuracy=best_accuracy,
          best_val_loss=best_val_loss,
          best_model_state=best_state,
          last_val_loss=last_val_loss,
          epochs_without_improvement=epochs_without_improvement,
          history=history,
          scaler=scaler,
          model_state_dict=best_state,
        )
        save_training_checkpoint(best_checkpoint_path, checkpoint_state)
    else:
      epochs_without_improvement += 1

    if should_save_checkpoints:
      save_training_checkpoint(
        last_checkpoint_path,
        _build_checkpoint_state(
          epoch=epoch,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          best_val_accuracy=best_accuracy,
          best_val_loss=best_val_loss,
          best_model_state=best_state,
          last_val_loss=last_val_loss,
          epochs_without_improvement=epochs_without_improvement,
          history=history,
          scaler=scaler,
        ),
      )

    if epochs_without_improvement >= patience:
      if should_log:
        log_message(
          config,
          (
            f'{fold_prefix}early stopping at epoch {epoch} | '
            f'best_val_acc={best_accuracy:.6f}'
          ),
        )
      break

  if best_state is not None:
    model.load_state_dict(best_state)

  if should_log:
    log_message(
      config,
      (
        f'{fold_prefix}training finished | '
        f'best_val_acc={best_accuracy:.6f} | '
        f'best_val_loss={best_val_loss:.6f} | '
        f'last_val_loss={last_val_loss:.6f}'
      ),
    )

  return best_accuracy, last_val_loss
