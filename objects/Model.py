from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier


MODELS = {
  'regression': lambda params: LogisticRegression(**params),
  'KNN': lambda params: KNeighborsClassifier(**params),
  'decision_tree': lambda params: DecisionTreeClassifier(**params),
  'random_forest': lambda params: RandomForestClassifier(**params),
  'catboost': lambda params: CatBoostClassifier(**params),
  'lightgbm': lambda params: LGBMClassifier(**params),
  'xgboost': lambda params: XGBClassifier(**params),
  'ensemble': None,
}

ENSEMBLE_MODELS = {
  'voting': VotingClassifier,
  'stacking': StackingClassifier,
}


def _get_enabled_ensemble_model_names(config) -> list[str]:
  selected = []
  for model_name, enabled in config.models_params.ensemble.models.items():
    if enabled:
      selected.append(model_name)

  return selected


def __get_model_name(config, name: str = ''):
  model_name = name if name else config.training_model
  
  if model_name not in MODELS:
    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS)}")

  return model_name

def get_model(config, name: str = '', params: dict | None = None):
  name = __get_model_name(config, name)
  
  params = params if params is not None else config.models_params[name]
  
  if name == 'ensemble':
    selected_models = _get_enabled_ensemble_model_names(config)
    estimators = [
      (model_name, MODELS[model_name](config.models_params[model_name]))
      for model_name in selected_models
    ]

    ensemble_type = config.models_params.ensemble.type
    ensemble_params = {}
    if ensemble_type == 'voting':
      ensemble_params['voting'] = config.models_params.ensemble.get('voting', 'hard')
      ensemble_params['weights'] = config.models_params.ensemble.get('weights', None)

    model = ENSEMBLE_MODELS[ensemble_type](
      estimators=estimators,
      **ensemble_params,
    )
  else:
    model = MODELS[name](params)
  
  return model
