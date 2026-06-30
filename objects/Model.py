from sklearn.linear_model import (
  ElasticNet,
  Lasso,
  LinearRegression,
  LogisticRegression,
  LogisticRegressionCV,
  Ridge,
  RidgeCV,
  RidgeClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
  RandomForestClassifier,
  RandomForestRegressor,
  StackingClassifier,
  StackingRegressor,
  VotingClassifier,
  VotingRegressor,
)
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


MODELS = {
  'regression': {
    'regression': lambda params: LinearRegression(**params),
    'ridge': lambda params: Ridge(**params),
    'lasso': lambda params: Lasso(**params),
    'elasticnet': lambda params: ElasticNet(**params),
    'KNN': lambda params: KNeighborsRegressor(**params),
    'decision_tree': lambda params: DecisionTreeRegressor(**params),
    'random_forest': lambda params: RandomForestRegressor(**params),
    'catboost': lambda params: CatBoostRegressor(**params),
    'lightgbm': lambda params: LGBMRegressor(**params),
    'xgboost': lambda params: XGBRegressor(**params),
    'voting': None,
    'stacking': None,
  },
  'classification': {
    'regression': lambda params: LogisticRegression(**params),
    'ridge': lambda params: RidgeClassifier(**params),
    'lasso': lambda params: LogisticRegression(**params),
    'elasticnet': lambda params: LogisticRegression(**params),
    'KNN': lambda params: KNeighborsClassifier(**params),
    'decision_tree': lambda params: DecisionTreeClassifier(**params),
    'random_forest': lambda params: RandomForestClassifier(**params),
    'catboost': lambda params: CatBoostClassifier(**params),
    'lightgbm': lambda params: LGBMClassifier(**params),
    'xgboost': lambda params: XGBClassifier(**params),
    'voting': None,
    'stacking': None,
  },
}

ENSEMBLE_MODELS = {
  ('regression', 'voting'): VotingRegressor,
  ('regression', 'stacking'): StackingRegressor,
  ('classification', 'voting'): VotingClassifier,
  ('classification', 'stacking'): StackingClassifier,
}


def _get_models_registry(config) -> dict:
  model_type = config.model_type
  if model_type not in MODELS:
    raise ValueError(
      f"Unknown model_type: {model_type!r}. Available: {list(MODELS)}"
    )
  return MODELS[model_type]


def _parse_ensemble_models(
  config,
  ensemble_name: str,
) -> tuple[list[str], list[float]]:
  names = []
  weights = []
  for model_name, weight in config.models_params[ensemble_name].models.items():
    weight = float(weight)
    if weight <= 0:
      continue
    names.append(model_name)
    weights.append(weight)

  if not names:
    raise ValueError(
      f'{ensemble_name} must include at least one model with weight > 0'
    )

  return names, weights


def _get_model_name(config, name: str = '') -> str:
  model_name = name if name else config.training_model
  models_registry = _get_models_registry(config)

  if model_name not in models_registry:
    raise ValueError(
      f"Unknown model: {model_name}. Available: {list(models_registry)}"
    )

  return model_name


def _build_model(config, name: str, params: dict | None = None):
  models_registry = _get_models_registry(config)
  params = dict(params if params is not None else config.models_params[name])
  return models_registry[name](params)


def _get_stacking_final_estimator(config):
  if config.model_type == 'regression':
    return RidgeCV()
  return LogisticRegressionCV()


def _build_ensemble(config, name: str):
  """Build voting or stacking ensemble from weighted base models in config."""
  selected_models, weights = _parse_ensemble_models(config, name)
  estimators = [
    (
      model_name,
      _build_model(config, model_name, config.models_params[model_name]),
    )
    for model_name in selected_models
  ]

  ensemble_params = {}
  if name == 'voting':
    ensemble_params['weights'] = weights
    voting_type = config.models_params.voting.get('voting')
    if voting_type is not None and config.model_type == 'classification':
      ensemble_params['voting'] = voting_type
  elif name == 'stacking':
    ensemble_params['final_estimator'] = _get_stacking_final_estimator(config)

  ensemble_cls = ENSEMBLE_MODELS[(config.model_type, name)]
  return ensemble_cls(estimators=estimators, **ensemble_params)


def get_model(config, name: str = '', params: dict | None = None):
  """Instantiate a sklearn model or ensemble selected by config.model_type."""
  name = _get_model_name(config, name)

  if name in {'voting', 'stacking'}:
    return _build_ensemble(config, name)

  params = params if params is not None else config.models_params[name]
  return _build_model(config, name, params)
