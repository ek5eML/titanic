import pandas as pd

from config import config
from objects.DataLoader import DataLoader
from objects.Trainer import Trainer
from utils import set_seed, log_run, save_best_model_params, save_model, load_model


def fit(config):
  set_seed(config.general.seed)
  trainer = Trainer(config)
  
  path_to_model = config.paths.path_to_checkpoints + f'/{config.training_model}.joblib'
  
  if config.mode == 'fit':
    res = trainer.fit_full()
    save_model(res, path_to_model)
  
  elif config.mode == 'train':
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

  elif config.mode == 'submit':
    data_loader = DataLoader(config)
    test_data = data_loader.load_test()
    
    model = load_model(path_to_model)
    predictions = model.predict(test_data)
    
    submission = pd.DataFrame({
      config.data.id_col: test_data.index,
      config.data.target_col: predictions.astype(int),
    })
    submission.to_csv(config.paths.path_to_submission, index=False)

if __name__ == '__main__':
  fit(config)
