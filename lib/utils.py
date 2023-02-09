from datetime import date
from typing import *
import tensorflow as tf
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math
import torch

class PairDataset(Dataset):
  """
  A PyTorch Dataset subclass to store (input, label) pair

  Parameters
  ----------
  inputs: iterable object
    The input dataset.

  labels: iterable object
    The label for input accordingly.
  """
  def __init__(self, inputs, labels):
    self.inputs = inputs
    self.labels = labels

  def __len__(self):
    """Length of the dataset."""
    return len(self.labels)

  def __getitem__(self, idx):
    """Get item of dataset, given index idx."""
    return self.inputs[idx], self.labels[idx]

def listify(o):
  """Make a list from the input object."""
  if o is None:
    return []
  if isinstance(o, list):
    return o
  if isinstance(o, str):
    return [o]
  if isinstance(o, Iterable):
    return list(o)
  return [o]

def compose(x, funcs, *args, order_key='_order', **kwargs):
  """Apply a list of funcs to input x in ascending order of order_key."""
  key = lambda o: getattr(o, order_key, 0)
  for f in sorted(listify(funcs), key=key):
    x = f(x, **kwargs)
  return x

def compute_metrics(true_df, pred_df, metric='rmse'):
  """
  Compute metrics for true values and predictions.

  Parameters
  ----------
  true_df: 2D array or DataFrame shape of (n_samples, n_columns)
    True array values.

  pred_df: 2D array or DataFrame shape of (n_samples, n_columns)
    Prediction values.

  metric: str, options: ['rmse', 'mae', 'mape'], default: 'rmse'
    Metric name to compute.

  Returns
  -------
  output: array shape of (n_columns,)
    Metric values for each column in the array.

  output_avg: float
    Average value of the metric, element-wise.
  """
  if isinstance(true_df, pd.DataFrame):
    true_df = true_df.values
  if isinstance(pred_df, pd.DataFrame):
    pred_df = pred_df.values

  if metric == 'rmse':
    f = tf.metrics.mean_squared_error
  elif metric == 'mae':
    f = tf.metrics.mean_absolute_error
  elif metric == 'mape':
    f = tf.metrics.mean_absolute_percentage_error
  elif metric == 'r2':
    f = tf.metrics.r_square.RSquare()

  output = []
  n = len(true_df)

  #(날짜, ) -> RNN
  if len(true_df.shape) == 1:
    y_true = true_df
    y_pred = pred_df
    m = f(y_true, y_pred).numpy()
    output.append(m)
  
  # (날짜, 지역) -> GNN
  else:
    for i in range(true_df.shape[1]):
      y_true = true_df[:, i]
      y_pred = pred_df[:, i]
      m = f(y_true, y_pred).numpy()
      output.append(m)
  
  output = np.array(output)
  output_avg = (output * n).sum() / (n * len(output))

  if metric == 'rmse':
    output = np.sqrt(output)
    output_avg = math.sqrt(output_avg)
  
  return output, output_avg

def save_predictions(y_pred,
                     model_name,
                     n_exp,
                     columns=None,
                     index=None,
                     path=''):
  """Save predictions according to model_name, n_exp in path directory."""
  df = pd.DataFrame(y_pred,
                    columns=columns,
                    index=index)
  df.to_csv(path + '/{}_pred_{}.csv'.format(model_name, str(n_exp)),
            index=index is not None)

def save_metrics(metrics, model_name, metric_name, columns=None, path=''):
  """Save metrics according to model_name, metric_name in path directory."""
  df = pd.DataFrame(metrics, columns=columns)
  df.to_csv(path + '/{}_{}.csv'.format(model_name, metric_name),
            index=False)

def get_distance_in_km_between_earth_coordinates(c1, c2):
  """Compute distance in km between 2 coordinates."""
  lat1, lon1 = c1
  lat2, lon2 = c2
  dLat = np.radians(lat2-lat1)
  dLon = np.radians(lon2-lon1)
  lat1 = np.radians(lat1)
  lat2 = np.radians(lat2)
  a = np.sin(dLat/2) * np.sin(dLat/2) + np.sin(dLon/2) * np.sin(dLon/2) * np.cos(lat1) * np.cos(lat2)
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
  earth_radius_km = 6371
  return earth_radius_km * c

def get_adjacency_matrix_2(dist_mx, normalized_k=0.1):
  """
  Compute adjacency matrix for a distance matrix.

  Parameters
  ---------- 
  dist_mx: 2D array shape of (n_entries, n_entries)
    A distance matrix.
  
  normalized_k: float, default: 0.1
    Entries that become lower than normalized_k after normalization
    are set to zero for sparsity.
  
  Returns
  -------
  adj_mx: 2D array shape of (n_entries)
    Adjacency matrix for the distance matrix.
  """

  # Calculates the standard deviation as theta.
  distances = dist_mx[~np.isinf(dist_mx)].flatten()
  std = distances.std()
  adj_mx = np.exp(-np.square(dist_mx / std))
  # Make the adjacent matrix symmetric by taking the max.
  # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

  # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
  adj_mx[adj_mx < normalized_k] = 0
  return adj_mx

def get_normalized_adj(A):
  """
  Returns the degree normalized adjacency matrix.
  """
  A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
  D = np.array(np.sum(A, axis=1)).reshape((-1,))
  D[D <= 10e-5] = 10e-5    # Prevent infs
  diag = np.reciprocal(np.sqrt(D))
  A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                        diag.reshape((1, -1)))
  return A_wave

# def np_compute_metrics(true_arr, pred_arr, metric='rmse'):
#   mae = np.abs(np.subtract(true_arr, pred_arr))

#   if metric == 'mae':
#     return np.mean(mae)
#   if metric == 'rmse':
#     mse = np.square(mae)
#     return np.sqrt(np.mean(mse))



def matplotlib_plot_font(KOR):
  '''
  To solve Broken Korean Font
  
  Parameters
  ---------- 
  None
  
  Returns
  -------
  None
  
  '''
  if KOR==True:
    # plot에서 한글 폰트 깨지는 현상 해결!
    from matplotlib import font_manager, rc
    font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family = font)
  else:
    # import matplotlib as mat
    # import matplotlib.font_manager as fm
    # import matplotlib as mpl
    from matplotlib import font_manager, rc
    import sys
    sys.path
    font_path = '/opt/anaconda3/envs/cluster_venv/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/Times New Roman Font.ttf'
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

    
def save_figure_predict(GROUND_TRUTH, y_pred,
                        # len_train=None, len_val=None, len_test=None, n_his_pred=None,
                        region_dict=None, suptitle = None, date_split = None, legend=None,
                        MAE=None, RMSE=None,
                        MAE_total=None, RMSE_total=None,
                        PATH=None):
    '''
    GROUND_TRUTH : pd.DataFrame
    y_pred [pd.DataFrame, pd.DataFrame ...]
    '''
    import matplotlib.pyplot as plt
    from math import ceil
    
    # base figure 정의해주고
    
    for num in range(ceil(GROUND_TRUTH.shape[1]/20)):
        fig = plt.figure(figsize=(25,15), facecolor='white')
        
        # 20개 이상인 지역들을 다른 figure에 plot 하기 위해 범위를 나눠줌
        # if num == 0: # city
        if num == 1: # state
            range_ = range(20)
        elif 20*(num+1) > GROUND_TRUTH.shape[1]: # 마지막이 20의 배수보다 작으면
            range_ = range(20*num, GROUND_TRUTH.shape[1])
        else:
            range_ = range(20*num, 20*(num+1))
        # print(y_pred.shape[1])
        for i in range_: # 17 도시에 대해서 각 도
            if i<20:
                fig.add_subplot(4,5,i+1)
            elif i<40:
                fig.add_subplot(4,5,i-19)
            elif i<60:
                fig.add_subplot(4,5,i-39)
            elif i<80:
                fig.add_subplot(4,5,i-59)
            elif i<100:
                fig.add_subplot(4,5,i-79)
            elif i<120:
                fig.add_subplot(4,5,i-99)
            elif i<140:
                fig.add_subplot(4,5,i-119)
            elif i<160:
                fig.add_subplot(4,5,i-139)
            elif i<180:
                fig.add_subplot(4,5,i-159)
            elif i<200:
                fig.add_subplot(4,5,i-179)
            elif i<220:
                fig.add_subplot(4,5,i-199)
            elif i<240:
                fig.add_subplot(4,5,i-219)
                    
            plt.plot(list(GROUND_TRUTH.iloc[: ,i].values), '--')
            # plt.plot(list(y_pred.iloc[: ,i].values), 'g', linewidth=0.6)
            # plt.plot(pred_val[i], 'g', linewidth=0.6) # GNN result
            if y_pred is not None:
              for k, pred in enumerate(y_pred):
                # print(pred)
                plt.plot(list(pred.iloc[: ,i].values), '-.')#, linewidth=0.6) # y_pred
            
            title_ = f"{region_dict[i]}"
            plt.title(title_, fontsize=15)
            plt.legend(legend)
            xlabels = [i[5:] for i in list(GROUND_TRUTH.index)] # i[5:] yyyy-mm-dd -> mm-dd
            plt.xticks(ticks = [i for i in range(len(xlabels))], labels = xlabels, rotation=90)
            plt.yticks(fontsize = 15)
            plt.ylim([min(list(GROUND_TRUTH.iloc[: ,i].values))-3, max(list(GROUND_TRUTH.iloc[: ,i].values))+3])

        if MAE_total is not None:
          plt.suptitle(f'{suptitle} \n {MAE_total:.4f}, {RMSE_total:.4f}', fontsize=30)
        
        else:
          plt.suptitle(f'{suptitle}', fontsize=30)
        
        plt.tight_layout()
        plt.savefig(f"{PATH}/{suptitle}_{num}.png")
        
    
import os
import numpy as np
import torch
import torch.utils.data
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# fvscode-webview://1sf5u8cg9afbdsp66k8tt56h2te3p8k3gmopl7ol6ebdngq3ce4f/f2d2def7-ad29-4a0a-a295-59771c204970rom .metrics import masked_mape_np
from scipy.sparse.linalg import eigs
# from .metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse,masked_mae_test,masked_rmse_test


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


# def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
#     '''
#     Parameters
#     ----------
#     distance_df_filename: str, path of the csv file contains edges information

#     num_of_vertices: int, the number of vertices

#     Returns
#     ----------
#     A: np.ndarray, adjacency matrix

#     '''
#     if 'npy' in distance_df_filename:

#         adj_mx = np.load(distance_df_filename)

#         return adj_mx, None

#     else:

#         import csv

#         A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
#                      dtype=np.float32)

#         distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
#                             dtype=np.float32)

#         if id_filename:

#             with open(id_filename, 'r') as f:
#                 id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

#             with open(distance_df_filename, 'r') as f:
#                 f.readline()
#                 reader = csv.reader(f)
#                 for row in reader:
#                     if len(row) != 3:
#                         continue
#                     i, j, distance = int(row[0]), int(row[1]), float(row[2])
#                     A[id_dict[i], id_dict[j]] = 1
#                     distaneA[id_dict[i], id_dict[j]] = distance
#             return A, distaneA

#         else:

#             with open(distance_df_filename, 'r') as f:
#                 f.readline()
#                 reader = csv.reader(f)
#                 for row in reader:
#                     if len(row) != 3:
#                         continue
#                     i, j, distance = int(row[0]), int(row[1]), float(row[2])
#                     A[i, j] = 1
#                     distaneA[i, j] = distance
#             return A, distaneA


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


# def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
#     '''
#     这个是为PEMS的数据准备的函数
#     将x,y都处理成归一化到[-1,1]之前的数据;
#     每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
#     该函数会把hour, day, week的时间串起来；
#     注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
#     这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
#     :param graph_signal_matrix_filename: str
#     :param num_of_hours: int
#     :param num_of_days: int
#     :param num_of_weeks: int
#     :param DEVICE:
#     :param batch_size: int
#     :return:
#     three DataLoaders, each dataloader contains:
#     test_x_tensor: (B, N_nodes, in_feature, T_input)
#     test_decoder_input_tensor: (B, N_nodes, T_output)
#     test_target_tensor: (B, N_nodes, T_output)

#     '''

#     file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

#     dirpath = os.path.dirname(graph_signal_matrix_filename)

#     filename = os.path.join(dirpath,
#                             file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_astcgn'

#     print('load file:', filename)

#     file_data = np.load(filename + '.npz')
#     train_x = file_data['train_x']  # (10181, 307, 3, 12)
#     train_x = train_x[:, :, 0:1, :]
#     train_target = file_data['train_target']  # (10181, 307, 12)

#     val_x = file_data['val_x']
#     val_x = val_x[:, :, 0:1, :]
#     val_target = file_data['val_target']

#     test_x = file_data['test_x']
#     test_x = test_x[:, :, 0:1, :]
#     test_target = file_data['test_target']

#     mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
#     std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

#     # ------- train_loader -------
#     train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

#     train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

#     # ------- val_loader -------
#     val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

#     val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # ------- test_loader -------
#     test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

#     test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # print
#     print('train:', train_x_tensor.size(), train_target_tensor.size())
#     print('val:', val_x_tensor.size(), val_target_tensor.size())
#     print('test:', test_x_tensor.size(), test_target_tensor.size())

#     return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std


# def compute_val_loss_mstgcn(net, val_loader, criterion,  masked_flag,missing_value,sw, epoch, limit=None):
#     '''
#     for rnn, compute mean loss on validation set
#     :param net: model
#     :param val_loader: torch.utils.data.utils.DataLoader
#     :param criterion: torch.nn.MSELoss
#     :param sw: tensorboardX.SummaryWriter
#     :param global_step: int, current global_step
#     :param limit: int,
#     :return: val_loss
#     '''

#     net.train(False)  # ensure dropout layers are in evaluation mode

#     with torch.no_grad():

#         val_loader_length = len(val_loader)  # nb of batch

#         tmp = []  # 记录了所有batch的loss

#         for batch_index, batch_data in enumerate(val_loader):
#             encoder_inputs, labels = batch_data
#             # print(encoder_inputs.shape)
#             outputs = net(encoder_inputs)
#             if masked_flag:
#                 loss = criterion(outputs, labels, missing_value)
#             else:
#                 loss = criterion(outputs, labels)

#             tmp.append(loss.item())
#             if batch_index % 100 == 0:
#                 print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
#             if (limit is not None) and batch_index >= limit:
#                 break

#         validation_loss = sum(tmp) / len(tmp)
#         sw.add_scalar('validation_loss', validation_loss, epoch)
#     return validation_loss


# def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method,_mean, _std, params_path, type):
#     '''

#     :param net: nn.Module
#     :param data_loader: torch.utils.data.utils.DataLoader
#     :param data_target_tensor: tensor
#     :param epoch: int
#     :param _mean: (1, 1, 3, 1)
#     :param _std: (1, 1, 3, 1)
#     :param params_path: the path for saving the results
#     :return:
#     '''
#     net.train(False)  # ensure dropout layers are in test mode

#     with torch.no_grad():

#         data_target_tensor = data_target_tensor.cpu().numpy()

#         loader_length = len(data_loader)  # nb of batch

#         prediction = []  # 存储所有batch的output

#         input = []  # 存储所有batch的input

#         for batch_index, batch_data in enumerate(data_loader):

#             encoder_inputs, labels = batch_data

#             input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)

#             outputs = net(encoder_inputs)

#             prediction.append(outputs.detach().cpu().numpy())

#             if batch_index % 100 == 0:
#                 print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

#         input = np.concatenate(input, 0)

#         # input = re_normalization(input, _mean, _std)

#         prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

#         print('input:', input.shape)
#         print('prediction:', prediction.shape)
#         print('data_target_tensor:', data_target_tensor.shape)
#         output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
#         np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

#         # 计算误差
#         excel_list = []
#         prediction_length = prediction.shape[2]

#         for i in range(prediction_length):
#             assert data_target_tensor.shape[0] == prediction.shape[0]
#             print('current epoch: %s, predict %s points' % (global_step, i))
#             if metric_method == 'mask':
#                 mae = masked_mae_test(data_target_tensor[:, :, i], prediction[:, :, i],0.0)
#                 rmse = masked_rmse_test(data_target_tensor[:, :, i], prediction[:, :, i],0.0)
#                 mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
#             else :
#                 mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
#                 rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
#                 mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
#             print('MAE: %.2f' % (mae))
#             print('RMSE: %.2f' % (rmse))
#             print('MAPE: %.2f' % (mape))
#             excel_list.extend([mae, rmse, mape])

#         # print overall results
#         if metric_method == 'mask':
#             mae = masked_mae_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
#             rmse = masked_rmse_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
#             mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
#         else :
#             mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
#             rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
#             mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
#         print('all MAE: %.2f' % (mae))
#         print('all RMSE: %.2f' % (rmse))
#         print('all MAPE: %.2f' % (mape))
#         excel_list.extend([mae, rmse, mape])
#         print(excel_list)