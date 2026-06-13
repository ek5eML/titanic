from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from objects.FeatureTransformer import FeatureTransformer
from objects.Model import get_model


def build_pipeline(config, name: str = '', params: dict | None = None):
  steps = [
    ('feature_transformer', FeatureTransformer(config)),
  ]

  if config.need_scaler:
    steps.append(('scaler', StandardScaler()))

  steps.append(('model', get_model(config, name, params)))

  return Pipeline(steps=steps)
