from stgraph_trainer.datasets import load_province_temporal_data
from stgraph_trainer.datasets import load_province_coordinates
from stgraph_trainer.datasets import preprocess_data_for_stgnn
from stgraph_trainer.utils import PairDataset
from stgraph_trainer.utils import compute_metrics
from stgraph_trainer.utils import matplotlib_plot_font
from stgraph_trainer.utils import save_figure_predict
from stgraph_trainer.datasets import Data2Graph
from torch.utils.data import DataLoader
from stgraph_trainer.models import TGCN, TGCNConv
from stgraph_trainer.trainers import TGCNTrainer
import torch
import numpy as np
import pandas as pd
import dgl
import scipy.sparse as sp

torch.manual_seed(42)
matplotlib_plot_font()

MODEL_NAME = "TGCN"
TIME_STEPS = 5
BATCH_SIZE = 16
EPOCHS = 50
learning_rate = 1e-3
device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

region_type = 'city'
df = pd.read_csv(f'/Users/jeonjunhwi/문서/Projects/Master_GNN/Data/KCDC_data/Processing_Results/smoothing_3_{region_type}_mean.csv', index_col=0, encoding='cp949')
df = df.iloc[100:707] # 델타 : 554, 540, 533 오미크론 : 707, 693, 686 / 2022여름 927
# df = df.iloc[:340] #340 326 319

region_dict = {}
for i, region in enumerate(df.columns):
    region_dict[i] = region 

#####################
## TEST START DATE ##
#####################    
split_date = '2021-11-25' #'2022-07-02' #'2021-11-25' #'2021-06-25' 
val_ratio = 0.2

train, val, test, _, _, scaler = preprocess_data_for_stgnn(data = df,
                                                           val_ratio=val_ratio,
                                                           split_date=split_date,
                                                           time_steps=TIME_STEPS)
X_train, y_train = train[0], train[1]
X_val, y_val = val[0], val[1]
X_test, y_test = test[0], test[1]


X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
n_test_samples = len(y_test)

train_dl = DataLoader(PairDataset(X_train, y_train),
                      batch_size=BATCH_SIZE,
                      shuffle=True)

val_dl = DataLoader(PairDataset(X_val, y_val),
                      batch_size=BATCH_SIZE,
                      shuffle=False)

test_dl = DataLoader(PairDataset(X_test, y_test),
                      batch_size=1,
                      shuffle=False)


dist_mx = pd.read_csv(f'data/distances_kr_{region_type}_adj_mx.csv', encoding='cp949', index_col=0)
# norm = 0.7

network_dict = dict({'corr' : [0.3, 0.5, 0.7, 0.9],
                     'corr-dist' : [0.3, 0.5, 0.7, 0.9],
                     'dist-corr' : [0.3, 0.5, 0.7, 0.9],
                     'dist_01' : [0.3, 0.5, 0.7, 0.9],
                     'dist_02' : [1, 2, 3],
                     'complete' : [0]})

MAE_total_list, MAPE_total_list, RMSE_total_list, idx_list = [], [], [], []

for network in network_dict.keys():
    for norm in network_dict[network]:
        data2network = Data2Graph(distance_matrix = dist_mx, temporal_data = df)
        G, adj_mx, graph_type = data2network.make_network(network_type=network,
                                                            region_type=region_type,
                                                            norm=norm,
                                                            int_adj=False) # 대체로 False가 더 좋았음.

        data2network.save_graph_html(enc=region_dict, title=region_type, save_name=f'{graph_type}_{norm}')

        # Save Graph Diameter
        import networkx as nx
        G1 = G.to_networkx()
        pd.DataFrame({'degree' : nx.degree_centrality(G1).values(),
                    'eigenvector' : nx.eigenvector_centrality(nx.Graph(G1), max_iter=1000).values(),
                    #   'katz' : nx.katz_centrality(nx.Graph(G1), max_iter=3000).values(),
                    'closeness' : nx.closeness_centrality(nx.Graph(G1)).values(),
                    'betweness' : nx.betweenness_centrality(nx.Graph(G1)).values(),
                    'clustering_coeff' : nx.clustering(nx.Graph(G1)).values()},
                    index= dist_mx.index
                    ).to_csv(f"Result/summary/{MODEL_NAME}_{graph_type}_{norm}_diameter.csv"
                        ,encoding='cp949')
                    
        date_split = f"{df.index[1]} ~ {df.index[len(train[0])]} ~ {df.index[len(train[0])+TIME_STEPS + len(val[0])+TIME_STEPS*2]} ~ {df.index[-1]}"
        suptitle_ = f"{MODEL_NAME}_{graph_type}_{norm}"
        save_path = f"save_model/{MODEL_NAME}_{graph_type}_{norm}_{df.index[1]} ~ {df.index[len(train[0])]} ~ {df.index[len(train[0])+TIME_STEPS + len(val[0])+TIME_STEPS*2]} ~.pt"

        model = TGCNConv(adj_mx = adj_mx,
                         hidden_dim=64,
                         out_dim=64,
                         num_hop=2)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
        trainer = TGCNTrainer(model = model,
                            train_loader = train_dl,
                            val_loader = val_dl,
                            test_loader = X_test.reshape(-1, len(df.columns),TIME_STEPS),
                            loss = loss_func,
                            optimizer = optimizer,
                            scaler = scaler,
                            device = device,
                            save_path = save_path,
                            length = len(df.columns),
                            TIME_STEPS = TIME_STEPS,
                            callbacks=None,
                            raw_test=df.iloc[-(n_test_samples + 1):].values)

        trainer.train(epochs=EPOCHS)
        y_pred = trainer.predict()

        df2 = pd.DataFrame(y_pred,
                    columns=df.columns,
                    index=df.iloc[-n_test_samples:].index)
        df2.to_csv(f'Result/pred/pred_TGCN_{graph_type}_{norm}.csv', encoding='cp949')

        # Compute RMSE of test dataset
        RMSE, RMSE_total = compute_metrics(df.iloc[-n_test_samples:], y_pred, metric='rmse')
        MAE, MAE_total = compute_metrics(df.iloc[-n_test_samples:], y_pred, metric='mae')
        MAPE, MAPE_total = compute_metrics(df.iloc[-n_test_samples:], y_pred, metric='mape')

        df.iloc[-n_test_samples:].to_csv('Result/pred/ground_truth.csv', encoding='cp949')

        matplotlib_plot_font()
        save_figure_predict(df, y_pred,
                            None, None, len(test[0]), TIME_STEPS,
                            region_dict, suptitle_,date_split,
                            MAE, MAPE, RMSE,
                            MAE_total, MAPE_total, RMSE_total, 
                        'Result/')
        
        MAE_total_list.append(MAE_total)
        MAPE_total_list.append(MAPE_total)
        RMSE_total_list.append(RMSE_total)
        idx_list.append(f'{network}_{norm}')
        
# Save for Total Metric of Each Metric
pd.DataFrame({'MAE' : MAE_total_list,
              'MAPE' : MAPE_total_list,
              'RMSE' : RMSE_total_list}, index=idx_list).to_csv(f'Result/summary/{MODEL_NAME}_total_metric.csv', encoding='cp949')