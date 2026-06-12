from config import config
from objects.Trainer import Trainer
from utils import set_seed, log_run, save_best_model_params


def fit(config):
  set_seed(config.general.seed)
  trainer = Trainer(config)
  
  if config.mode == 'fit':
    res = trainer.fit_full()
    
  res = trainer.run_cv()
  
  if config.logging:
    log_run(
      config=config,
      model_name=res['model_name'],
      model_params=res['model_params'],
      metric_name=config.metric,
      metric_value=res['metric'],
      metric_std=res['metric_std'],
      time_s=res['time_s'],
    )
  
  if config.save_best_model:
    save_best_model_params(
      registry_name=config.training_model,
      model_params=res['model_params'],
      metric_name=config.metric,
      metric_value=res['metric'],
      metric_std=res['metric_std'],
    )

if __name__ == '__main__':
  fit(config)
