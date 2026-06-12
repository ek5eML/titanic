import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
) -> tuple[float, dict]:
  device = _get_device(device_name)
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  best_accuracy = 0.0
  best_state = None
  epochs_without_improvement = 0
  last_val_loss = 0.0

  for _ in range(num_epochs):
    train_epoch(model, train_loader, criterion, optimizer, device)
    last_val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

    if val_accuracy > best_accuracy:
      best_accuracy = val_accuracy
      best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
      }
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1
      if epochs_without_improvement >= patience:
        break

  if best_state is not None:
    model.load_state_dict(best_state)

  training_info = {
    'best_val_accuracy': best_accuracy,
    'last_val_loss': last_val_loss,
  }
  return best_accuracy, training_info
