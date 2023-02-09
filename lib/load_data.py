import pandas as pd
import os
from .utils import listify
import torch
import numpy as np

def load_df(filename):
  """Load dataframe from a csv file with `filename` in `data` directory."""
  module_path = os.path.dirname(__file__)
  return pd.read_csv(module_path + '/data/' + filename)

def load_province_temporal_data(provinces='Seoul', status='New'):
  """Load temporal data for province(s) and status from original data file."""
  filename = '101_DT_COVID19_temporal_data_eng.csv'
  df = load_df(filename)
  df = df.drop(['Item', 'Unit'], axis=1)
  provinces = listify(provinces)
  temporal_data = {}
  for p in provinces:
    temporal_data[p] = df[(df['Category'] == p) & (df['Status'] == status)]\
      .iloc[0, 2:].tolist()
  return pd.DataFrame(temporal_data,
                      index=pd.to_datetime(df.columns[2:]),
                      dtype='float32')

def load_province_coordinates():
  """Load coordiantes of provinces."""
  filename = 'Region.csv'
  region_df = load_df(filename)
  province_coords = region_df[region_df.city == region_df.province][
    ['province', 'latitude', 'longitude']].iloc[:-1]
  # Swap Incheon and Gwangju
  temp_row = province_coords.iloc[3].copy()
  province_coords.iloc[3] = province_coords.iloc[4].copy()
  province_coords.iloc[4] = temp_row
  return province_coords.reset_index(drop=True)

def load_weather_data(provinces='Seoul',
                      columns=['Average_Temp',
                               'Average_wind_speed',
                               'Average_Humidity'],
                      start_date='2020-04-01',
                      end_date='2021-01-12'):
  """Load weather data for province(s)."""
  filename = 'OBS_ASOS_DD_20210426110338.csv'
  provinces = listify(provinces)
  columns = listify(columns)
  weather_df = load_df(filename)
  weather_df = weather_df.drop(['Daily_precipitation', 'Point'], axis=1)
  weather_df = weather_df[weather_df['Point_name'].isin(provinces)]
  weather_df['Time'] = pd.to_datetime(weather_df['Time'])
  weather_df = weather_df[(weather_df['Time'] >= start_date) & 
                          (weather_df['Time'] <= end_date)]
  weather_df = weather_df.reset_index(drop=True)
  weather_dfs = []
  for col in columns:
    temp_df = weather_df.loc[:, ['Point_name', 'Time', col]]
    dfs = []
    for p in provinces:
      p_data = temp_df[temp_df['Point_name'] == p].loc[:, ['Time', col]]
      p_data.reset_index(drop=True, inplace=True)
      p_data.set_index('Time', inplace=True)
      p_data.columns = [p]
      dfs.append(p_data)
    df = pd.concat(dfs, axis=1)
    df.fillna(method='bfill', inplace=True)
    weather_dfs.append(df)
  return weather_dfs

def load_cases_data(provinces='Seoul',
                    status='Domestic',
                    start_date='2020-04-09',
                    end_date='2021-04-25'):
  """Load case data for province(s)."""
  filename = 'cases.csv'
  provinces = listify(provinces)
  cases_df = load_df(filename)

  cases_df = cases_df[cases_df['Province'].isin(provinces)]
  cases_df['Time'] = pd.to_datetime(cases_df['Time'])

  cases_df = cases_df[(cases_df['Time'] >= start_date) & (cases_df['Time'] <= end_date)]
  cases_df = cases_df.reset_index(drop=True)
  
  temp_df = cases_df.loc[:, ['Province', 'Time', status]]
  dfs = []
  for p in provinces:
    p_data = temp_df[temp_df['Province'] == p].loc[:, ['Time', status]].reset_index(drop=True)
    p_data.set_index('Time', inplace=True)
    p_data.columns = [p]
    dfs.append(p_data)
  df = pd.concat(dfs, axis=1)

  return df

def load_data(file_path, len_train, len_val):
    df = pd.read_csv(file_path, header=None).values.astype(float)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    return train, val, test


def data_transform(data, n_his, n_pred, device):
    # produce data slices for training and testing
    n_route = data.shape[1] # 17개 도시에 대해서
    l = len(data)
    num = l-n_his#-n_pred
    x = np.zeros([num, 1, n_his, n_route])
    y = np.zeros([num, n_route])
    
    cnt = 0
    for i in range(num):
        head = i # i
        tail = i + n_his # i + 5
        # print(tail + n_pred - 1)
        # print(data[tail + n_pred - 1])
        x[cnt, :, :, :] = data[head: tail].reshape(1, n_his, n_route)
        y[cnt] = data[tail] #  + n_pred - 1
        cnt += 1
    # print(data)
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
# 이거 n_his 만큼 학습해서 y_pred만큼 예측하는것인가봄...

def seq_data(data,sequence_length, device):
    x_seq = []
    y_seq = []
    for i in range(len(data) - sequence_length):
        x_seq.append(data[i:i+sequence_length])
        y_seq.append(data[i+sequence_length])
        
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view(-1, 1)