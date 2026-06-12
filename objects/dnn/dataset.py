import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_dataloader(
  X: np.ndarray,
  y: np.ndarray,
  batch_size: int,
  shuffle: bool,
) -> DataLoader:
  X_tensor = torch.tensor(X, dtype=torch.float32)
  y_tensor = torch.tensor(y, dtype=torch.long)
  dataset = TensorDataset(X_tensor, y_tensor)

  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
