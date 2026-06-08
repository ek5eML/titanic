import pandas as pd


class DataLoader:
  def __init__(self, config):
    self.config = config
    
  def load_data(self, path: str) -> pd.DataFrame:
    '''
    Load the data by path.
    '''
    if self.config.data.id_col:
      df = pd.read_csv(path, index_col=self.config.data.id_col)
    else:
      df = pd.read_csv(path)
    
    return df
    
  def load_train(self) -> pd.DataFrame:
    '''
    Load the train data by path in config.
    '''
    return self.load_data(self.config.paths.path_to_train_data)
  
  def load_test(self) -> pd.DataFrame:
    '''
    Load the test data by path in config.
    '''
    return self.load_data(self.config.paths.path_to_test_data)
  
  def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    '''
    Split the data into features and target.
    '''
    X = df.drop(columns=[self.config.data.target_col])
    y = df[self.config.data.target_col]
    
    return X, y
