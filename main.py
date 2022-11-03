# import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import torch
import shutil

from lib import PairDataset
from lib import Data2Graph
from lib import compute_metrics
from lib import matplotlib_plot_font
from lib import save_figure_predict
from lib import Trainer
from lib import preprocess_data
from model_select import model_selection

matplotlib_plot_font()

MODEL_NAME = "GCN"
diff_ = '1st'
tmp = '_'+diff_ # custom folder name
TIME_STEPS = 5
BATCH_SIZE = 16
EPOCHS = 50
learning_rate = 1e-3
validation_ratio = 0.2
device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

# Load Data
region_type = 'city'
dataset_name = 'covid_alpha'
dist_mx = pd.read_csv(f'./Data/distances_kr_{region_type}_adj_mx.csv', encoding='cp949', index_col=0)

# data는 1로 고정
df = pd.read_csv(f'Data/KCDC_data/Processing_Results/smoothing_1_{region_type}_mean.csv', index_col=0, encoding='cp949')
df = df.iloc[1:707] # 델타 : 554, 540, 533 오미크론 : 707, 693, 686 / 402 ~ 20210226 백신
# df = df.iloc[:340] # 델타 : 554, 540, 533 오미크론 : 707, 693, 686 / 402 ~ 20210226 백신 ~402
region_dict = {}
for i, region in enumerate(df.columns):
    region_dict[i] = region 

#####################
## TEST START DATE ##
#####################    
split_date = '2021-11-25' #'2021-11-25' #'2021-06-25' 
# split_date = '2020-11-23' # df[:402]


train, val, test, scaler, horizon = preprocess_data(data = df,
                                                    val_ratio=validation_ratio,
                                                    split_date=split_date,
                                                    time_steps=TIME_STEPS,
                                                    diff_=diff_)

train_dl = DataLoader(PairDataset(train[0], train[1]), batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(PairDataset(val[0], val[1]), batch_size=BATCH_SIZE, shuffle=False)
test_dl = DataLoader(PairDataset(test[0], test[1]), batch_size=1, shuffle=False)
# [Batch, node, time, channel]

MAE_total_list, MAPE_total_list, RMSE_total_list, idx_list = [], [], [], []

network = 'dist_01'
norm = 0.5


result_dir = ('%s_%s_%s'+tmp) % (MODEL_NAME, network, norm)
model_dir = ('%s_%s_%s'+tmp) % (MODEL_NAME, network, norm)
Network_path = os.path.join('Result', dataset_name, 'Network', str(horizon), result_dir)
Figure_path = os.path.join('Result', dataset_name, 'Figure', str(horizon), result_dir)
Diameter_path = os.path.join('Result', dataset_name, 'Diameter', str(horizon), result_dir)
Pred_path = os.path.join('Result', dataset_name, 'Pred', str(horizon), result_dir)
model_path = os.path.join('Save_model', dataset_name, str(horizon), model_dir)

if os.path.exists(Network_path):
    shutil.rmtree(Network_path) # 최종 경로로 해야함
if os.path.exists(Figure_path):
    shutil.rmtree(Figure_path) # 해당 경로 데이터 모두 삭제
if os.path.exists(Diameter_path):
    shutil.rmtree(Diameter_path) # 해당 경로 데이터 모두 삭제
if os.path.exists(Pred_path):
    shutil.rmtree(Pred_path) # 해당 경로 데이터 모두 삭제
if os.path.exists(model_path):
    shutil.rmtree(model_path) # 해당 경로 데이터 모두 삭제
    
os.makedirs(Network_path) # 새로 폴더 생성
os.makedirs(Figure_path) # 새로 폴더 생성
os.makedirs(Diameter_path) # 새로 폴더 생성
os.makedirs(Pred_path) # 새로 폴더 생성
os.makedirs(model_path) # 새로 폴더 생성

data2network = Data2Graph(distance_matrix = dist_mx, temporal_data = df)
G, adj_mx, graph_type = data2network.make_network(network_type=network,
                                                region_type=region_type,
                                                norm=norm,
                                                int_adj=False,
                                                Diameter_path = Diameter_path)
data2network.save_graph_html(enc=region_dict,
                             title=region_type,
                             save_name=f'{graph_type}_{norm}',
                             Network_path = Network_path)

######################
# Save Graph Diameter#
######################
import networkx as nx
G1 = G.to_networkx()
pd.DataFrame({'degree' : nx.degree_centrality(G1).values(),
            'eigenvector' : nx.eigenvector_centrality(nx.Graph(G1), max_iter=1000).values(),
            'closeness' : nx.closeness_centrality(nx.Graph(G1)).values(),
            'betweness' : nx.betweenness_centrality(nx.Graph(G1)).values(),
            'clustering_coeff' : nx.clustering(nx.Graph(G1)).values()},
            index= dist_mx.index
            ).to_csv(f"{Diameter_path}/{MODEL_NAME}_{graph_type}_{norm}_diameter.csv",encoding='cp949')        
print('Confirm Symmetric(False is Symmetric) : ', False in (adj_mx.numpy() == adj_mx.numpy().transpose()))

###############
# Train Model #
###############
model = model_selection(MODEL_NAME = MODEL_NAME, adj_mx = adj_mx, TIME_STEPS = TIME_STEPS, device=device, save_path = model_path)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

trainer = Trainer(model=model,
                train_loader=train_dl,
                val_loader=val_dl,
                test_loader=test[0],
                adj=adj_mx,
                scaler=scaler,
                loss=loss_func,
                optimizer=optimizer,
                device=device,
                save_path=model_path,
                callbacks=None,
                raw_test=df.iloc[-(horizon + 1):].values)
trainer.train(EPOCHS)
y_pred= trainer.predict()

###################
# Save Prediction #
###################
df.iloc[-horizon:].to_csv(f'{Pred_path}/ground_truth.csv', encoding='cp949')
df2 = pd.DataFrame(
                   y_pred,
                   columns=df.columns,
                   index=df.iloc[-horizon:].index
                   ).to_csv(f'{Pred_path}/{graph_type}_{norm}.csv', encoding='cp949')

################################
# Compute RMSE of test dataset #
################################
RMSE, RMSE_total = compute_metrics(df.iloc[-horizon:], y_pred, metric='rmse')
MAE, MAE_total = compute_metrics(df.iloc[-horizon:], y_pred, metric='mae')
MAPE, MAPE_total = compute_metrics(df.iloc[-horizon:], y_pred, metric='mape')

###############
# save figure #
###############
suptitle = f"{MODEL_NAME}_{graph_type}_{norm}"
save_figure_predict(df, y_pred,
                    None, None, len(test[0]),
                    n_his_pred = TIME_STEPS,
                    region_dict = region_dict,
                    suptitle = suptitle,
                    date_split = f"{df.index[1]} ~ {df.index[len(train[0])]} ~ {df.index[len(train[0])+TIME_STEPS + len(val[0])+TIME_STEPS*2]} ~ {df.index[-1]}",
                    MAE = MAE, MAPE = MAPE, RMSE = RMSE,
                    MAE_total = MAE_total, MAPE_total = MAPE_total, RMSE_total = RMSE_total, 
                    PATH=Figure_path)

MAE_total_list.append(MAE_total)
MAPE_total_list.append(MAPE_total)
RMSE_total_list.append(RMSE_total)
idx_list.append(f'{network}_{norm}')

########################################
# Save for Total Metric of Each Metric #
########################################
pd.DataFrame({
            'MAE' : MAE,
            'MAPE' : MAPE,
            'RMSE' : RMSE,
            }, index=region_dict.values()).to_csv(f'{Pred_path}/{suptitle}_metric.csv', encoding='cp949')

pd.DataFrame({
            'MAE' : MAE_total_list,
            'MAPE' : MAPE_total_list,
            'RMSE' : RMSE_total_list
              }, index=idx_list).to_csv(f'{Pred_path}/{suptitle}_total_metric.csv', encoding='cp949')