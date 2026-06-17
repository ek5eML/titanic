from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


MODELS = {
  'regression': lambda params: LogisticRegression(**params),
  'KNN': lambda params: KNeighborsClassifier(**params),
  'decision_tree': lambda params: DecisionTreeClassifier(**params),
  'random_forest': lambda params: RandomForestClassifier(**params),
  'catboost': lambda params: CatBoostClassifier(**params),
  'lightgbm': lambda params: LGBMClassifier(**params),
  'xgboost': lambda params: XGBClassifier(**params),
}

def __get_model_name(config, name: str = ''):
  model_name = name if name else config.training_model
  
  if model_name not in MODELS:
    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS)}")

  return model_name

def get_model(config, name: str = '', params: dict | None = None):
  name = __get_model_name(config, name)
  
  params = params if params is not None else config.models_params[name]
  
  return MODELS[name](params)
