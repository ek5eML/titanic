import time

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from objects.DataLoader import DataLoader
from objects.FeatureTransformer import FeatureTransformer
from objects.dnn.checkpoint import (
  get_best_checkpoint_path,
  get_last_checkpoint_path,
  load_training_checkpoint,
)
from objects.dnn.dataset import make_dataloader
from objects.dnn.model import DNN
from objects.dnn.train_loop import _get_device, fit_model
from utils import (
  begin_log_section,
  fit_scale,
  get_cv_splitter,
  get_scaler,
  log_run,
  transform_scale,
)


class DNNRunner:
  """Train, cross-validate, and predict with the PyTorch DNN."""

  def __init__(self, config):
    self.config = config

  def _get_params(self, params: dict | None = None) -> dict:
    if params is not None:
      return dict(params)

    raw = self.config.models_params.DNN
    params = OmegaConf.to_container(raw, resolve=True)
    return dict(params) if params is not None else {}

  def _preprocess(
    self,
    X_train,
    X_val,
  ) -> tuple[np.ndarray, np.ndarray, object | None, FeatureTransformer]:
    """Fit feature transformer and optionally scale train/val arrays."""
    feature_transformer = FeatureTransformer(self.config)
    X_train = feature_transformer.fit_transform(X_train)
    X_val = feature_transformer.transform(X_val)

    scaler = get_scaler(self.config)
    X_train = fit_scale(scaler, X_train)
    X_val = transform_scale(scaler, X_val)

    return X_train, X_val, scaler, feature_transformer

  def fit_full(self, name = None, params: dict | None = None) -> dict:
    """Train DNN on train/val split with early stopping and checkpointing."""
    params = self._get_params(params)
    start = time.perf_counter()
    last_checkpoint_path = get_last_checkpoint_path(self.config)
    best_checkpoint_path = get_best_checkpoint_path(self.config)

    if self.config.rerun:
      for path in (last_checkpoint_path, best_checkpoint_path):
        if path.exists():
          path.unlink()

    resume = (
      not self.config.rerun
      and self.config.save_last_model
      and last_checkpoint_path.exists()
    )

    if self.config.logging:
      begin_log_section(self.config)

    data_loader = DataLoader(self.config)
    train_data = data_loader.load_train()
    X, y = data_loader.split_data(train_data)
    y = y.to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
      X,
      y,
      test_size=self.config.fit.val_size,
      random_state=self.config.general.seed,
      stratify=y if self.config.model_type == 'classification' else None,
    )

    X_train, X_val, scaler, feature_transformer = self._preprocess(X_train, X_val)

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
      batch_norm=params['batch_norm'],
    )
    best_val_metric, last_val_loss = fit_model(
      model,
      train_loader,
      val_loader,
      lr=params['lr'],
      weight_decay=params['weight_decay'],
      num_epochs=params['num_epochs'],
      patience=params['patience'],
      device_name=params.get('device', 'auto'),
      config=self.config,
      scaler=scaler,
      feature_transformer=feature_transformer,
      last_checkpoint_path=last_checkpoint_path,
      best_checkpoint_path=best_checkpoint_path,
      resume=resume,
    )

    elapsed_s = time.perf_counter() - start

    if self.config.logging:
      log_run(
        config=self.config,
        model_name='DNN',
        model_params=params,
        metric_name=self.config.metric,
        metric_value=best_val_metric,
        metric_std=0.0,
        time_s=elapsed_s,
        with_header=False,
      )

    return {
      'model': model,
      'scaler': scaler,
      'feature_transformer': feature_transformer,
      'model_params': params,
      'best_val_metric': best_val_metric,
      'last_val_loss': last_val_loss,
    }

  def run_cv(self, name: str = '', params: dict | None = None) -> dict:
    """Run manual CV by retraining the DNN on each fold."""
    params = self._get_params(params)
    data_loader = DataLoader(self.config)
    train_data = data_loader.load_train()
    X, y = data_loader.split_data(train_data)
    y = y.to_numpy()

    if self.config.logging:
      begin_log_section(self.config)

    cv = get_cv_splitter(self.config)
    fold_metrics: list[float] = []

    start = time.perf_counter()
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
      X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
      y_train, y_val = y[train_idx], y[val_idx]

      X_train, X_val, _, _ = self._preprocess(X_train, X_val)

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
        batch_norm=params['batch_norm'],
      )
      fold_metric, _ = fit_model(
        model,
        train_loader,
        val_loader,
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        num_epochs=params['num_epochs'],
        patience=params['patience'],
        device_name=params.get('device', 'auto'),
        config=self.config,
        fold=fold_idx,
      )
      fold_metrics.append(fold_metric)

    elapsed_s = time.perf_counter() - start
    fold_metrics_arr = np.array(fold_metrics)

    result = {
      'model_name': 'DNN',
      'model_params': params,
      'metric': float(fold_metrics_arr.mean()),
      'metric_std': float(fold_metrics_arr.std()),
      'fold_metrics': fold_metrics,
      'time_s': elapsed_s,
    }

    if self.config.logging:
      log_run(
        config=self.config,
        model_name=result['model_name'],
        model_params=result['model_params'],
        metric_name=self.config.metric,
        metric_value=result['metric'],
        metric_std=result['metric_std'],
        time_s=result['time_s'],
        with_header=False,
      )

    return result

  def predict(self, test_data) -> np.ndarray:
    """Load best DNN checkpoint and predict on preprocessed test features."""
    checkpoint_path = get_best_checkpoint_path(self.config)
    if not checkpoint_path.exists():
      raise FileNotFoundError(f'DNN checkpoint not found: {checkpoint_path}')

    checkpoint = load_training_checkpoint(checkpoint_path)
    params = self._get_params()
    scaler = checkpoint['scaler']
    feature_transformer = checkpoint.get('feature_transformer')
    if feature_transformer is None:
      raise ValueError(
        'DNN checkpoint is missing feature_transformer. '
        'Re-run fit with rerun=True before submit.'
      )

    X = feature_transformer.transform(test_data)
    X = transform_scale(scaler, X)

    model = DNN(
      n_features=X.shape[1],
      hidden_dims=list(params['hidden_dims']),
      dropout=params['dropout'],
      out_dim=self.config.general.num_classes,
      batch_norm=params['batch_norm'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    device = _get_device(params.get('device', 'auto'))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
      logits = model(torch.tensor(X, dtype=torch.float32, device=device))
      predictions = logits.argmax(dim=-1).cpu().numpy()

    return predictions
