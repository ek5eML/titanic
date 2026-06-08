from config import config
from objects.Trainer import Trainer
from utils import set_seed, log_run


def fit(config):
  set_seed(config.general.seed)
  trainer = Trainer(config)
  res = trainer.run_cv()
  
  if config.logging:
    log_run(
      config=config,
      model_name=res['model_name'],
      model_params=res['model_params'],
      metric_name=config.metric,
      metric_value=res['metric'],
      time_s=res['time_s'],
    )

if __name__ == '__main__':
  fit(config)
