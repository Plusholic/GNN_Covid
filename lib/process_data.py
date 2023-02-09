import pandas as pd
import numpy as np
from .utils import listify
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class WindowMinMaxScaler:
    
    def __init__(self, window):
        self.window = window
        
    def fit_transform(self, df):
        self.df = df
        self.max_df = pd.DataFrame({}, columns=self.df.columns, index=self.df.index)
        past = 0
        for _ in range(int(self.df.shape[0]/self.window)):
            
            # 해당 window 동안의 maximum 값을 저장
            self.max_df.iloc[past : past + self.window, :] = self.df.iloc[past : past + self.window, :].max()
            past = past + self.window

        print('processing_index : ', past)
        
        # 해당 window 동안 max 확진자가 0인곳은 0으로 나눠줘야하기 때문에 에러 발생
        # inverse transform 할 때도 해당 window동안 0이라면 1을 곱해줘도 0이기 때문에 1. 으로 할당
        self.max_df[self.max_df == 0] = 1.
        self.new_df = self.df / self.max_df
        
        return self.new_df.dropna().values

    def inverse_transform(self):
        return (self.new_df * self.max_df).dropna().values

def data_diff(data = None, diff = '1st'):
  """
  Calculates the difference of a Dataframe element compared with previous element in the Dataframe.

  Parameters
  ----------
  data: 2D array
    Input array data.

  Returns
  -------
  data: DataFrame
    Difference array data, or the changes to the observations from one to the next.
  """
  if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)
  
  if diff == '1st':
    print('1st difference')
    data = data.diff().dropna() # 1차 차분
  
  elif diff == 'log':
    print('1st difference after log tranform')
    data = np.log(data)
    data.replace([np.inf, -np.inf], 0, inplace=True) # inf to zero
    data = data.diff().dropna() # 로그 1차 차분
  
  elif diff == '2nd':
    print('2nd difference')
    data = data.diff().dropna() # 1차 차분
    data = data.diff().dropna() # 2차 차분
    
  elif diff == 'diff_smooth':
    smooth = pd.read_csv(f'Data/KCDC_data/Processing_Results/smoothing_5_city_mean.csv', index_col=0, encoding='cp949')
    tmp = pd.DataFrame(1, index=data.index, columns=data.columns)
    smooth = (tmp * smooth).fillna(0)
    data = data.iloc[:len(data)] - smooth.iloc[:len(data)]
    
  else:
    data = data
  
  # data = data.iloc[1:, :]
  return data


def timeseries_to_supervised(data, lag=2):
  """
  Turn time series data to supervised data for machine learning model.

  Parameters
  ----------
  data: 2D array shape of (n_samples, n_timeseries)
    The original time serires data. n_timeseires is the number of time series in the data.

  lags: int, defautl: 1
    The number of time steps (or context length) of the series to be processed to make prediction.

  Returns
  -------
  output: 3D array shape of (n_samples - lags, lags + 1, n_timeseries)
    The supervised sequence data for machine learning model.
  """
  if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)
  dfs = [data.shift(i).iloc[lag:].values for i in range(lag, -1, -1)]
  return np.array(dfs).transpose(1, 0, 2)
  
def preprocess_data(data,
                    val_ratio,
                    split_date,
                    time_steps,
                    diff_=None,
                    feature_range=None):
  """
  Prepare data for STGNN model.

  Parameters
  ----------
  data: 2D array shape of (n_samples, n_timeseries)
    The original time serires data. n_timeseires is the number of time series in the data.

  split_date: str, format of YYYY-MM-DD
    The date to split data into train and test set.

  time_steps: int
    The number of time steps (or context length) of the series to be processed to make prediction.

  feature_range: tuple, defautl: None
    The min, max value for using MinMaxScaler to normalize data.
    Default is None, which means using StandardScaler.

  Returns
  -------
  X_train: 3D array shape of (n_train_samples - time_steps, n_timeseries, time_steps)
    The supervised train data.

  y_train: 2D array shape of (n_train_samples - time_steps, n_timeseries)
    The supervised train labels.

  X_test: 3D array shape of (n_test_samples, n_timeseries, time_steps)
    The supervised test data.

  y_test: 2D array shape of (n_test_samples, n_timeseries)
    The supervised test labels.

  train: 2D array shape of (n_train_samples, n_timeseries)
    The value of train dataset, after differencing.

  test: 2D array shape of (n_test_samples + time_steps, n_timeseries)
    The value of test dataset, after differencing.

  scaler: object
    MinMaxScaler or Standard Scaler used to transform the data.
  """
  
  if diff_ is not None:
    data = data_diff(data = data, diff = diff_) # 전날과의 차이를 학습
  
  train_df = data[data.index < split_date]
  test = data.iloc[len(train_df):, :]
  
  len_val = int(train_df.shape[0] * val_ratio)
  len_train = train_df.shape[0] - len_val
  train = train_df.iloc[:len_train,:]
  val = train_df.iloc[len_train:,:]

  print('train : ', train.index[-1])
  print('val : ', val.index[-1])
  print('test : ', test.index[-1])
  
  if feature_range is not None:
    scaler = MinMaxScaler(feature_range=feature_range)
  else:
    scaler = StandardScaler()

  train_scaled = scaler.fit_transform(train)
  val_scaled = scaler.transform(val)
  test_scaled = scaler.transform(test)
  
  train_arr = timeseries_to_supervised(train_scaled, lag=time_steps)
  X_train = train_arr[:, :-1, :].transpose(0, 2, 1)
  y_train = train_arr[:, -1, :]

  val_arr = timeseries_to_supervised(val_scaled, lag=time_steps)
  X_val = val_arr[:, :-1, :].transpose(0, 2, 1)
  y_val = val_arr[:, -1, :]

  test_arr = timeseries_to_supervised(test_scaled, lag=time_steps)
  X_test = test_arr[:, :-1, :].transpose(0, 2, 1)
  y_test = test_arr[:, -1, :]
  
  import torch
  X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
  y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
  X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
  y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
  X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
  y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
  horizon = len(y_test)
  
  return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, horizon