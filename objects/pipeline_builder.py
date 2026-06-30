from sklearn.pipeline import Pipeline

from objects.FeatureTransformer import FeatureTransformer
from objects.Model import get_model
from utils import get_scaler


def build_pipeline(config, name: str = '', params: dict | None = None):
  """Build sklearn pipeline: features -> optional scaler -> model."""
  steps = [
    ('feature_transformer', FeatureTransformer(config)),
  ]

  scaler = get_scaler(config)
  if scaler is not None:
    steps.append(('scaler', scaler))

  steps.append(('model', get_model(config, name, params)))

  return Pipeline(steps=steps)
