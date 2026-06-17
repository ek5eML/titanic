from objects.dnn.runner import DNNRunner
from objects.sklearn.runner import SklearnRunner


def get_runner(config):
  if config.training_model == 'DNN':
    return DNNRunner(config)

  return SklearnRunner(config)

class Trainer:
  def __init__(self, config):
    self.config = config
    self.runner = get_runner(config)

  def run_cv(self, name: str = '', params: dict | None = None):
    return self.runner.run_cv(name, params)

  def fit_full(self, name: str = '', params: dict | None = None):
    return self.runner.fit_full(name, params)

  def predict(self, test_data):
    return self.runner.predict(test_data)
