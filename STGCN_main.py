import dgl # huggingface 같은 라이브러리
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from stgraph_trainer.datasets import *
from stgraph_trainer.trainers import STGCNTrainer
from stgraph_trainer.utils import save_figure_predict
from stgraph_trainer.utils import matplotlib_plot_font
from stgraph_trainer.models import STGCN_WAVE
import torch.nn as nn
import scipy.sparse as sp

import os
print(os.getcwd())

matplotlib_plot_font()
torch.manual_seed(42)
# device = torch.device("mps")
device = torch.device("cpu")

MODEL_NAME = "STGCN_1"
#######################################
## Load Data and Train, Test Setting ##
#######################################
region_type = 'state'
df2 = pd.read_csv(f'/Users/jeonjunhwi/문서/Projects/Master_GNN/Data/KCDC_data/Processing_Results/smoothing_3_{region_type}_mean.csv', index_col=0, encoding='cp949')
# df = df.iloc[100:660]
# df2 = df

df = df2.iloc[100:700] # 12월 까지만 해보자
df2 = df2.iloc[100:700] # 아래쪽 plot을 위해서 저장해준 원래부분, 깔끔하게 바꾸자

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
len_val = int(df.shape[0] * 0.1)
len_test = 24
len_train = df.shape[0] - len_val - len_test

train = df.iloc[: len_train,:]
val = df.iloc[len_train: len_train + len_val,:]
test = df.iloc[len_train + len_val:len_train + len_val + len_test,:]

print('train date : ~', train.index[-1], len(train) ,'days')
print('val date : ~', val.index[-1], len(val) ,'days')
print('test date : ~', test.index[-1] ,len(test) ,'days')
# input("ctrl + c")
date_split = f"{train.index[0]}~{train.index[-1]}~{val.index[-1]}~{test.index[-1]}"

############################
## Hyperparameter Setting ##
############################
n_his = 5
save_path = 'stgcnwavemodel.pt'
control_str = 'TSNT'#TNTST' #'TSNT'
num_layers = len(control_str)
n_pred = 1
n_route = num_nodes
blocks = [1, 32, 64, 128, 32, 128]
# blocks = [1, 32, 128, 64, 32, 128]
drop_prob = 0
batch_size = 16
epochs = 50
lr = 0.001


# from stgraph_trainer.datasets import Data2Graph
# import pandas as pd
##########################
## Make Network Setting ##
##########################
region_type = 'state'
graph_type = f'dist_01_{region_type}'
dist_mx = pd.read_csv(f'data/distances_kr_{region_type}_adj_mx.csv', encoding='cp949', index_col=0)
norm = 0.9
data2network = Data2Graph(distance_matrix = dist_mx, temporal_data = df)
G, adj_mx, graph_type = data2network.make_network(network_type='dist_01',
                                                    region_type=region_type,
                                                    norm=norm,
                                                    int_adj=True)

###########################
## Save Network Diameter ##
###########################
import networkx as nx
G1 = G.to_networkx()
pd.DataFrame({'degree' : nx.degree_centrality(G1).values(),
              'eigenvector' : nx.eigenvector_centrality(nx.Graph(G1), max_iter=300).values(),
              'katz' : nx.katz_centrality(nx.Graph(G1)).values(),
              'closeness' : nx.closeness_centrality(nx.Graph(G1)).values(),
              'betweness' : nx.betweenness_centrality(nx.Graph(G1)).values(),
              'clustering_coeff' : nx.clustering(nx.Graph(G1)).values()},
             index= dist_mx.index
             ).to_csv(f"Result/summary/{MODEL_NAME}_{graph_type}_{norm}_diameter.csv"
                 ,encoding='cp949')

##########################################
## Data Scaling and Generate Dataloader ##
##########################################

scaler = StandardScaler()
suptitle_ = f"{MODEL_NAME}_{graph_type}_{norm}_{control_str}"
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

###################
## Model Setting ##
###################

loss = nn.MSELoss()
G = G.to(device)
model = STGCN_WAVE(blocks, n_his, n_route, G, drop_prob, num_layers, device, control_str).to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)


trainer = STGCNTrainer(model=model,
                       train_loader=train_loader,
                       val_loader=val_loader,
                    #    test_loader=test_loader,
                       test_loader=torch.tensor(x_test, dtype=torch.float32), # x_test
                       loss=loss,
                       optimizer=optimizer,
                       scaler=scaler,
                       device=device,
                       save_path=save_path,
                       scheduler=scheduler,
                       raw_test=df2.iloc[-(len(y_test) + 1):].values)

trainer.train(epochs)
# sequence_length = 5

y_pred = trainer.predict(df.shape[1])
print(y_pred.shape)
print(df2.iloc[-(len(y_test)):].shape)

#######################
### Save Prediction ###
#######################

from stgraph_trainer.utils import compute_metrics
RMSE, RMSE_total = compute_metrics(df2.iloc[-len(y_test):], y_pred, metric='rmse')
MAE, MAE_total = compute_metrics(df2.iloc[-len(y_test):], y_pred, metric='mae')
MAPE, MAPE_total = compute_metrics(df2.iloc[-len(y_test):], y_pred, metric='mape')

pd.DataFrame(y_pred,
             columns=[region_dict[i] for i in range(df2.shape[1])],
             index = df2.iloc[-len(y_test):].index
             ).to_csv(f"Result/pred/pred_STGCN_{graph_type}_{norm}.csv",encoding='cp949')

save_figure_predict(df2, y_pred,
                    len_train, len_val, len_test, n_his,
                    region_dict, suptitle_,date_split,
                    MAE, MAPE, RMSE,
                    MAE_total, MAPE_total, RMSE_total, 
                    'Result/')

print(f'MAE : {MAE_total}, MAPE : {MAPE_total}, RMSE : {RMSE_total}')

