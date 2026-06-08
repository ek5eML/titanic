from sklearn.pipeline import Pipeline

from objects.FeatureTransformer import FeatureTransformer
from objects.Model import get_model


def build_pipeline(config, name: str = '', params: dict | None = None):
  return Pipeline(steps=[
    ('feature_transformer', FeatureTransformer(config)),
    ('model', get_model(config, name, params)),
  ])