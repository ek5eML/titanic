from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


TITLES_DICT = {
 'Mlle': 'Miss',
 'Mme': 'Miss',
 'Ms': 'Miss',
 'Dr': 'Mr',
 'Major': 'Mr',
 'Sir': 'Mr',
 'Capt': 'Mr',
 'Don': 'Mr',
 'Lady': 'Mrs',
 'the Countess': 'Mrs',
 'Jonkheer': 'Other',
 'Col': 'Other',
 'Rev': 'Other',
}

AGE_BY_TITLE_DICT = {
    'Master': 4.57,
    'Miss': 21.86,
    'Mr': 32.74,
    'Mrs': 36,
    'Other': 46,
}

AGE_BINS = [
  [0.419, 20],
  [20.0, 26.0],
  [26.0, 32.74],
  [32.74, 38.0],
  [38.0, 80.0],
]

FARE_BINS = [
  [-0.001, 7.854],
  [7.854, 10.5],
  [10.5, 21.679],
  [21.679, 39.688],
  [39.688, 512.329],
]

EMBARKED_DICT = {
    'S': 0,
    'C': 1,
    'Q': 2,
}

DROP_COLUMNS = [
    'Name', 'Ticket', 'Cabin',
    'Age', 'Fare',
    'SibSp', 'Parch',
]

def generate_continuous_feature(
    data: pd.DataFrame,
    base_col_name: str,
    cat_col_name: str,
    bins: list[list[float]],
) -> pd.DataFrame:
    data = data.copy()
    
    for i, (lo, hi) in enumerate(bins):
        data.loc[(data[base_col_name] > lo) & (data[base_col_name] <= hi), cat_col_name] = i

    data.loc[data[base_col_name] <= bins[0][1], cat_col_name] = 0
    data.loc[data[base_col_name] > bins[-1][0], cat_col_name] = i

    data[cat_col_name] = data[cat_col_name].astype('int64')

    return data

class FeatureTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, config):
    self.config = config
  
  def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
    titles = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).replace(TITLES_DICT)
    df['Age'] = df['Age'].fillna(titles.map(AGE_BY_TITLE_DICT))
    
    df['Embarked'] = df['Embarked'].fillna('S')
    
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    return df
      
  def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df['Embarked'] = df['Embarked'].replace(EMBARKED_DICT).astype('int64')
    df['Sex'] = (df['Sex'] == 'female').astype('int64')
    df = generate_continuous_feature(df, 'Age', 'AgeBin', AGE_BINS)
    df = generate_continuous_feature(df, 'Fare', 'FareBin', FARE_BINS)    
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = (df['FamilySize'] == 0).astype('int64')

    return df
    
  def fit(self, X: pd.DataFrame, y = None):
    return self

  def transform(self, X):
    df = X.copy()
    
    df = self._fill_missing(df)
    df = self._build_features(df)
    df = df.drop(DROP_COLUMNS, axis=1)

    return df