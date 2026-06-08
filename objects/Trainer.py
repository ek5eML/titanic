import time

from sklearn.model_selection import cross_validate, StratifiedKFold

from objects.pipeline_builder import build_pipeline
from objects.DataLoader import DataLoader


class Trainer:
  def __init__(self, config):
    self.config = config
    
  def run_cv(self, name: str = '', params: dict | None = None):
    data_loader = DataLoader(self.config)
    train_data = data_loader.load_train()
    X, y = data_loader.split_data(train_data)
    
    pipeline = build_pipeline(self.config, name, params)
    
    cv = StratifiedKFold(**self.config.cv)

    start = time.perf_counter()
    res = cross_validate(
      pipeline,
      X, y,
      cv=cv,
      scoring=self.config.metric,
    )
    elapsed_s = time.perf_counter() - start
    model = pipeline.named_steps['model']

    return {
      'model_name': model.__class__.__name__,
      'model_params': model.get_params(),
      'metric': res['test_score'].mean(),
      'metric_std': res['test_score'].std(),
      'fold_metrics': res['test_score'].tolist(),
      'time_s': elapsed_s,
    }
  
  def fit_full(self, name: str = '', params: dict | None = None):
    pass
    