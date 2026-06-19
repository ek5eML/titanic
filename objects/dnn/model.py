import torch
import torch.nn as nn


class DNN(nn.Module):
  def __init__(
    self,
    n_features: int,
    hidden_dims: list[int],
    dropout: float,
    out_dim: int = 1,
    batch_norm: bool = False,
  ):
    super().__init__()
    layers: list[nn.Module] = []
    in_dim = n_features

    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(in_dim, hidden_dim))
      if batch_norm:
        layers.append(nn.BatchNorm1d(hidden_dim))
      layers.append(nn.ReLU())
      if dropout > 0:
        layers.append(nn.Dropout(dropout))
      in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, out_dim))
    self.layers = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.layers(x)
    if logits.shape[-1] == 1:
      return logits.squeeze(-1)
    return logits
