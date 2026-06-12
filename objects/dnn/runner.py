import time

import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from objects.DataLoader import DataLoader
from objects.FeatureTransformer import FeatureTransformer
from objects.dnn.dataset import make_dataloader
from objects.dnn.model import DNN
from objects.dnn.train_loop import fit_model


class DNNRunner:
  def __init__(self, config):
    self.config = config

  def _get_params(self) -> dict:
    raw = self.config.models_params.DNN
    params = OmegaConf.to_container(raw, resolve=True)
    return dict(params) if params is not None else {}

  def run_cv(self) -> dict:
    params = self._get_params()
    data_loader = DataLoader(self.config)
    train_data = data_loader.load_train()
    X, y = data_loader.split_data(train_data)
    y = y.to_numpy()

    cv = StratifiedKFold(**self.config.cv)
    fold_metrics: list[float] = []

    start = time.perf_counter()
    for train_idx, val_idx in cv.split(X, y):
      X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
      y_train, y_val = y[train_idx], y[val_idx]

      feature_transformer = FeatureTransformer(self.config)
      X_train = feature_transformer.fit_transform(X_train)
      X_val = feature_transformer.transform(X_val)

      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train.to_numpy(dtype=np.float32))
      X_val = scaler.transform(X_val.to_numpy(dtype=np.float32))

      train_loader = make_dataloader(
        X_train, y_train, params['batch_size'], shuffle=True,
      )
      val_loader = make_dataloader(
        X_val, y_val, params['batch_size'], shuffle=False,
      )

      model = DNN(
        n_features=X_train.shape[1],
        hidden_dims=list(params['hidden_dims']),
        dropout=params['dropout'],
        out_dim=self.config.general.num_classes,
      )
      fold_accuracy, _ = fit_model(
        model,
        train_loader,
        val_loader,
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        num_epochs=params['num_epochs'],
        patience=params['patience'],
        device_name=params.get('device', 'auto'),
      )
      fold_metrics.append(fold_accuracy)

    elapsed_s = time.perf_counter() - start
    fold_metrics_arr = np.array(fold_metrics)

    return {
      'model_name': 'DNN',
      'model_params': params,
      'metric': float(fold_metrics_arr.mean()),
      'metric_std': float(fold_metrics_arr.std()),
      'fold_metrics': fold_metrics,
      'time_s': elapsed_s,
    }
