from omegaconf import OmegaConf

config = {
  'general': {
    'experiment_name': 'baseline',
    'seed': 31415,
    'num_classes': 2,
  },
  'paths': {
    'path_to_train_data': './data/train.csv',
    'path_to_test_data': './data/test.csv',
    'path_to_checkpoints': './checkpoints/${general.experiment_name}',
    'path_to_logs': './logs/${general.experiment_name}',
  },
  'data': {
    'target_col': 'Survived',
    'id_col': 'PassengerId',
  },
  'cv': {
    'n_splits': 5,
    'shuffle': True,
    'random_state': '${general.seed}',
  }, 
  'models_params': {
    'log_reg': {
      'C': 0.1,
      'class_weight': None,
      'l1_ratio': 0.9,
      'max_iter': 100,
      'solver': 'saga',
    },
    'KNN': {},
    'decision_tree': {},
    'random_forest': {},
    'catboost': {},
    'lightgbm': {},
    'xgboost': {},
    'DNN': {},
  },
  'logging': True,
  'rerun': False,
  'mode': 'train', # непонятно пока надо или нет
  'task_type': 'classification', # непонятно пока надо или нет
  'training_model': 'log_reg',
  'metric': 'accuracy',
}

config = OmegaConf.create(config)
