from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from objects.FeatureTransformer import FeatureTransformer
from objects.Model import get_model


def build_pipeline(config, name: str = '', params: dict | None = None):
  steps = [
    ('feature_transformer', FeatureTransformer(config)),
  ]
  
  if params.get('need_scaler', False):
    steps.append(('scaler', StandardScaler()))

  model_params = {k: v for k, v in params.items() if k != 'need_scaler'}  

  steps.append(('model', get_model(config, name, model_params)))
  
  return Pipeline(steps=steps)
