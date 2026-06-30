import pandas as pd


class DataLoader:
  """Load train/test CSV files and split features from target."""

  def __init__(self, config):
    self.config = config

  def load_data(self, path: str) -> pd.DataFrame:
    """Read a CSV file and optionally use id_col as index."""
    if self.config.data.id_col:
      df = pd.read_csv(path, index_col=self.config.data.id_col)
    else:
      df = pd.read_csv(path)

    return df

  def load_train(self) -> pd.DataFrame:
    """Load the training dataset."""
    return self.load_data(self.config.paths.path_to_train_data)

  def load_test(self) -> pd.DataFrame:
    """Load the test dataset."""
    return self.load_data(self.config.paths.path_to_test_data)

  def get_combined_features(self) -> pd.DataFrame:
    """Concatenate train and test features without target for global preprocessing."""
    train_df = self.load_train()
    X_train = train_df.drop(columns=[self.config.data.target_col])
    X_test = self.load_test()

    return pd.concat([X_train, X_test], axis=0)

  def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X and y using config target column."""
    X = df.drop(columns=[self.config.data.target_col])
    y = df[self.config.data.target_col]

    return X, y
