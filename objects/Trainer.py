def get_runner(config):
  """Return sklearn or DNN runner based on config.training_model."""
  if config.training_model == 'DNN':
    from objects.dnn.runner import DNNRunner
    return DNNRunner(config)

  from objects.sklearn.runner import SklearnRunner
  return SklearnRunner(config)


class Trainer:
  """Facade that delegates training, CV, and prediction to a model runner."""

  def __init__(self, config):
    self.config = config
    self.runner = get_runner(config)

  def run_cv(self, name: str = '', params: dict | None = None):
    """Run cross-validation for the selected model."""
    return self.runner.run_cv(name, params)

  def fit_full(self, name: str = '', params: dict | None = None):
    """Train the model on a train/val split and save artifacts."""
    return self.runner.fit_full(name, params)

  def predict(self, test_data):
    """Generate predictions for test features."""
    return self.runner.predict(test_data)
