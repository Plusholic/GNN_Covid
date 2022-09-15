from operator import index
from posixpath import supports_unicode_filenames
import dgl # huggingface 같은 라이브러리
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from processing.data_scaler import *
from processing.load_data import *
from trainers.STGCN_trainer import STGCNTrainer
from utils.utils import *
from models.model import *
import torch.nn as nn
from processing.sensors2graph import *
import scipy.sparse as sp

import os
print(os.getcwd())

matplotlib_plot_font()
torch.manual_seed(42)
# device = torch.device("mps")
device = torch.device("cpu")

#######################################
## Load Data and Train, Test Setting ##
#######################################
region_type = 'state'
df2 = pd.read_csv(f'/Users/jeonjunhwi/문서/Projects/Master_GNN/Data/KCDC_data/Processing_Results/smoothing_3_{region_type}_mean.csv', index_col=0, encoding='cp949')
# df = df.iloc[100:660]
# df2 = df

df = df2.iloc[100:660] # 12월 까지만 해보자
df2 = df2.iloc[100:660] # 아래쪽 plot을 위해서 저장해준 원래부분, 깔끔하게 바꾸자
df = df.diff()
df = df.iloc[1:, :]
# print(df)
# input(" ")
### 컬럼을 숫자로 바꿔줌 ###
region_dict = {}
for i, region in enumerate(df.columns):
    region_dict[i] = region

num_samples, num_nodes = df.shape
print("number of nodes:",num_nodes)
print("number of samples:",num_samples)

# 시계열 데이터이기 때문에 랜덤 샘플링 하지 않음 훈련 데이터 세트는 과거, 검증 데이터 세트는 항상 약간 뒤의 미래
# W = adj_mx
len_val = 25
len_test = 24
len_train = df.shape[0] - len_val - len_test

train = df[: len_train]
val = df[len_train: len_train + len_val]
test = df[len_train + len_val:len_train + len_val + len_test]
# print(len(train), len(val), len(test))
# print('train date : ~', train.index[-1])
# print('val date : ~', val.index[-1])
# print('test date : ~', test.index[-1])
# print(" ")
train = df.iloc[: len_train,:]
val = df.iloc[len_train: len_train + len_val,:]
test = df.iloc[len_train + len_val:len_train + len_val + len_test,:]
# print(len(train), len(val), len(test))
# print('train date : ~', train.index[-1])
# print('val date : ~', val.index[-1])
# print('test date : ~', test.index[-1])
# input("ctrl + c")


############################
## Hyperparameter Setting ##
############################
n_his = 5
save_path = 'stgcnwavemodel.pt'
control_str = 'TSNT'#TNTST' #'TSNT'
num_layers = len(control_str)
n_pred = 1
n_route = num_nodes
blocks = [1, 64, 128, 128, 32, 128]
# blocks = [1, 32, 128, 64, 32, 128]
drop_prob = 0
batch_size = 16
epochs = 50
lr = 0.001

#########################
## Distance Network 01 ##
# #######################
# norm = 0.9
# graph_type = f'dist_01_{region_type}_{train.index[0]}~{test.index[-1]}'
# dist_mx = pd.read_csv(f'/Users/jeonjunhwi/문서/Projects/Master_GNN/stgcn_wave/data/sensor_graph/distances_kr_{region_type}_adj_mx.csv', encoding='cp949', index_col=0)
    
# adj_mx = get_adjacency_matrix(dist_mx, normalized_k=norm) #Generates the adjacent matrix from the distance between sensors and the sensor ids. 
# sp_mx = sp.coo_matrix(adj_mx)
# G = dgl.from_scipy(sp_mx)
# pd.DataFrame({'region' : dist_mx.columns, 'degree' : G.in_degrees()}).to_csv('Result/summary/degree.csv', encoding='cp949')

#########################
## Distance Network 02 ##
# #######################
# graph_type = f'dist_02_{region_type}_{train.index[0]}~{test.index[-1]}'
# # distance_df = pd.read_csv('/Users/jeonjunhwi/문서/Projects/Master_GNN//stgcn_wave/data/sensor_graph/distances_kr_metro_city.csv', dtype={'from': 'str', 'to': 'str'}, index_col=0)
# dist_mx = pd.read_csv(f'/Users/jeonjunhwi/문서/Projects/Master_GNN/stgcn_wave/data/sensor_graph/distances_kr_{region_type}_adj_mx.csv', encoding='cp949', index_col=0)
# norm = 1
# G, adj_mx = make_dist_network(dist_mx, threshold=norm)
# sp_mx = sp.coo_matrix(adj_mx)
# G = dgl.from_scipy(sp_mx)

#########################
## Correlation Network ##
#########################
norm = 0.9
graph_type = f'test_minus_{region_type}_{train.index[0]}~{test.index[-1]}'
# daily_df_state = pd.read_csv('/Users/jeonjunhwi/문서/Projects/Master_GNN/stgcn_wave/data/smoothing_3_state_mean.csv', encoding="euc-kr",index_col =0)
G, adj_mx = make_corr_network(df, threshold=norm, minus_mean=True)
G = dgl.from_networkx(G)

# ####################################
# ### node degree, graph html 저장 ###
# ####################################
pd.DataFrame({'region' : df.columns, 'degree' : G.in_degrees()}).to_csv(f'Result/summary/{graph_type}_degree.csv', encoding='cp949')
save_graph_html(adj_mx = adj_mx,
                enc = region_dict,
                title = graph_type + ' ' + str(norm),
                save_name = f'Result/html/state_network_vis_{graph_type}_{norm}')
# input("Press Enter after See Network File ")

##########################################
## Data Scaling and Generate Dataloader ##
##########################################

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = FluctuationScaler(df)
# scaler = LogScaler(log_category='log10')
suptitle_ = f"Standard_{graph_type}_{norm}_{control_str}_NAdam_GCN2"
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

# tensor data로 변환
x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)
input(" ctrl + c : ")
# DataLoader 생성
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
# for x, y in train_loader:
#     print(x.shape, y.shape)

# print(train_loader)
# print(val_loader)
# print(test_loader)
loss = nn.MSELoss()
G = G.to(device)

model = STGCN_WAVE(blocks, n_his, n_route, G, drop_prob, num_layers, device, control_str).to(device)


optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)


trainer = STGCNTrainer(model=model,
                       train_loader=train_loader,
                       val_loader=val_loader,
                       test_loader=test_loader,
                       loss=loss,
                       optimizer=optimizer,
                       scaler=scaler,
                       device=device,
                       save_path=save_path,
                       scheduler=scheduler,
                       raw_test=df2.iloc[-(len(y_test) + 1):].values)

trainer.train(epochs)
# sequence_length = 5

gru_res = pd.DataFrame({})
lstm_res = pd.DataFrame({})

(MAE, MAPE, RMSE), (MAE_total, MAPE_total, RMSE_total), y_pred = trainer.predict(df.shape[1])

gru_res, lstm_res = None, None


pd.DataFrame(y_pred.numpy(), columns=[i for i in range(df2.shape[1])]).to_csv(f"data/{graph_type}_{norm}.csv")
save_figure_predict(df2, y_pred,
                    gru_res, lstm_res,
                    len_train, len_val, len_test, n_his,
                    region_dict, suptitle_,
                    MAE, MAPE, RMSE,
                    MAE_total, MAPE_total, RMSE_total, 
                    'Result/')

print(f'MAE : {MAE_total}, MAPE : {MAPE_total}, RMSE : {RMSE_total}')

