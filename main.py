# import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import torch
import shutil
import numpy as np

from lib import PairDataset
from lib import Data2Graph
from lib import compute_metrics
from lib import matplotlib_plot_font
from lib import save_figure_predict
from lib import Trainer
from lib import preprocess_data
from model_select import model_selection
torch.manual_seed(2022)
np.random.seed(2022)
matplotlib_plot_font()

# MODEL_NAME = "GCN"
model_list = ['STGCN']#, 'GCN2', 'STGNN', 'STGCN', 'ASTGCN', 'TGCN', 'DCRNN']
diff_ = '1st'
# network = 'dist_02'
# norm = 2
tmp = '_' + str(diff_) # custom folder name
TIME_STEPS = 5
BATCH_SIZE = 16
EPOCHS = 100
learning_rate = 1e-2
validation_ratio = 0.2
dataset_name = 'meeting'
device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')


# Load Data
region_type = 'city'
dist_mx = pd.read_csv(f'./Data/distances_kr_{region_type}_adj_mx.csv', encoding='cp949', index_col=0)

# data는 1로 고정
df = pd.read_csv(f'Data/KCDC_data/Processing_Results/smoothing_1_{region_type}_mean.csv', index_col=0, encoding='cp949')
df = df.iloc[400:707] # 델타 : 554, 540, 533 오미크론 : 707, 693, 686 / 402 ~ 20210226 백신
# df = df.iloc[:340] # 델타 : 554, 540, 533 오미크론 : 707, 693, 686 / 402 ~ 20210226 백신 ~402
region_dict = {}
for i, region in enumerate(df.columns):
    region_dict[i] = region 

network_dict = dict({
                        # 'corr' : [0.5, 0.7],
                    #  'corr-dist' : [0.5, 0.7],
                    #  'dist-corr' : [0.7, 0.9],
                     'cross_corr' : [0.5],
                    #  'cross_corr-dist' : [0.3, 0.4],
                    #  'dist-cross_corr' : [0.7, 0.9],
                    #  'dist_01' : [0.7, 0.9],
                    #  'dist_02' : [1, 2, 3],
                    #  'complete' : [0],
                    #  'identity' : [0],
                     })
region_MAE=pd.DataFrame({},index=df.columns)
region_RMSE=pd.DataFrame({},index=df.columns)
total_metric=pd.DataFrame({})
for MODEL_NAME in model_list:
    for network in network_dict.keys():
        for norm in network_dict[network]:



            #####################
            ## TEST START DATE ##
            #####################    
            split_date = '2021-11-25' #'2021-11-25' #'2021-06-25' 
            # split_date = '2020-11-23' # df[:402]


            train, val, test, scaler, horizon = preprocess_data(data = df,
                                                                val_ratio=validation_ratio,
                                                                split_date=split_date,
                                                                time_steps=TIME_STEPS,
                                                                diff_=diff_)#,
                                                                # feature_range = (0, 1))

            train_dl = DataLoader(PairDataset(train[0], train[1]), batch_size=BATCH_SIZE, shuffle=True)
            val_dl = DataLoader(PairDataset(val[0], val[1]), batch_size=BATCH_SIZE, shuffle=False)
            test_dl = DataLoader(PairDataset(test[0], test[1]), batch_size=1, shuffle=False)
            # [Batch, node, time, channel]

            MAE_total_list, MAPE_total_list, RMSE_total_list, idx_list = [], [], [], []

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

            data2network = Data2Graph(distance_matrix = dist_mx, temporal_data = df) # 여기서 df.diff()를 썼어야하나?
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
            # pd.DataFrame({'degree' : nx.degree_centrality(G1).values(),
            #             'eigenvector' : nx.eigenvector_centrality(nx.Graph(G1), max_iter=1000).values(),
            #             'closeness' : nx.closeness_centrality(nx.Graph(G1)).values(),
            #             'betweness' : nx.betweenness_centrality(nx.Graph(G1)).values(),
            #             'clustering_coeff' : nx.clustering(nx.Graph(G1)).values()},
            #             index= dist_mx.index
            #             ).to_csv(f"{Diameter_path}/{MODEL_NAME}_{graph_type}_{norm}_diameter.csv",encoding='cp949')        
            print('Confirm Symmetric(False is Symmetric) : ', False in (adj_mx.numpy() == adj_mx.numpy().transpose()))

            ###############
            # Train Model #
            ###############
            model = model_selection(MODEL_NAME = MODEL_NAME,
                                    adj_mx = adj_mx,
                                    TIME_STEPS = TIME_STEPS,
                                    device=device,
                                    save_path = model_path)

            loss_func = torch.nn.MSELoss() #L1Loss
            optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,70], gamma=0.1)

            trainer = Trainer(model=model,
                            train_loader=train_dl,
                            val_loader=val_dl,
                            test_loader=test[0],
                            adj=adj_mx,
                            scaler=scaler,
                            loss=loss_func,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            device=device,
                            save_path=model_path,
                            callbacks=None,
                            res_df = df)
                            # raw_test=df.iloc[-(horizon + 1):].values)
            trainer.train(EPOCHS)
            y_pred= trainer.predict(diff_, horizon)

            ###################
            # Save Prediction #
            ###################
            
            if diff_ == '1st':
                ## 1st diff
                GROUND_TRUTH = df.diff().iloc[-horizon:,]# + df.iloc[-(horizon+1):-1,].values
                PRED = y_pred# + df.iloc[-(horizon+1):-1,].values
                # PRED_SM = y_pred + smooth.iloc[-(horizon+1):-1,].values
                
            elif diff_ == '2nd':
                ## 2nd diff
                res_ = 2*df.diff().iloc[-(horizon+1):-1].values + df.iloc[-(horizon+2):-2].values
                # res_sm = 2*smooth.diff().iloc[-(horizon+1):-1].values + smooth.iloc[-(horizon+1):-1].values

                GROUND_TRUTH = df.diff().diff().iloc[-horizon:,] + res_
                PRED = y_pred + res_
                # PRED_SM = y_pred + res_sm

            elif diff_ == 'log':
                ## log diff
                data = np.log(df)
                data.replace([np.inf, -np.inf], 0, inplace=True) # inf to zero
                # smooth_ = np.log(smooth)
                # smooth_.replace([np.inf, -np.inf], 0, inplace=True) # inf to zero

                GROUND_TRUTH = np.exp(data.diff().iloc[-horizon:,] + data.iloc[-(horizon+1):-1].values)
                PRED = np.exp(y_pred + data.iloc[-(horizon+1):-1].values)
                # PRED_SM = np.exp(y_pred + smooth_.iloc[-(horizon+1):-1].values)
            
            else:
                GROUND_TRUTH = df.iloc[-horizon:]
                PRED = y_pred.iloc[-horizon:]
                # PRED_SM = smooth.iloc[-horizon:]
            
            GROUND_TRUTH.to_csv(f'{Pred_path}/ground_truth.csv', encoding='cp949')
            PRED.to_csv(f'{Pred_path}/{graph_type}_{norm}_predict.csv', encoding='cp949')
            # PRED_SM.to_csv(f'{Pred_path}/{graph_type}_{norm}_smpred.csv', encoding='cp949')
            ################################
            # Compute RMSE of test dataset #
            ################################
            RMSE, RMSE_total = compute_metrics(GROUND_TRUTH, y_pred, metric='rmse')
            MAE, MAE_total = compute_metrics(GROUND_TRUTH, y_pred, metric='mae')
            # MAPE, MAPE_total = compute_metrics(GROUND_TRUTH, y_pred, metric='mape')

            ###############
            # save figure #
            ###############
            suptitle = f"{MODEL_NAME}_{graph_type}_{norm}"
            save_figure_predict(
                                df = GROUND_TRUTH,
                                y_pred = PRED,
                                test_data = None, #PRED_SM,# dataframe이니까 value로 바꿔서 넣어야함.
                                region_dict = region_dict,
                                suptitle = suptitle,
                                legend = ['GROUND TRUTH', MODEL_NAME, 'Test_data'],
                                date_split = f"{df.index[1]} ~ {df.index[len(train[0])]} ~ {df.index[len(train[0])+TIME_STEPS + len(val[0])+TIME_STEPS*2]} ~ {df.index[-1]}",
                                MAE = MAE, RMSE = RMSE,
                                MAE_total = MAE_total, RMSE_total = RMSE_total, 
                                PATH=Figure_path
                                )

            suptitle = f"{MODEL_NAME}_{graph_type}_{diff_}_{norm}"
            save_figure_predict(
                                df = GROUND_TRUTH+df.iloc[-(horizon+1):-1,].values,
                                y_pred = PRED+df.iloc[-(horizon+1):-1,].values,
                                test_data = None,
                                region_dict = region_dict,
                                suptitle = suptitle,
                                legend = ['GROUND TRUTH', MODEL_NAME, 'Test_data'],
                                date_split = f"{df.index[1]} ~ {df.index[len(train[0])]} ~ {df.index[len(train[0])+TIME_STEPS + len(val[0])+TIME_STEPS*2]} ~ {df.index[-1]}",
                                MAE = MAE, RMSE = RMSE,
                                MAE_total = MAE_total, RMSE_total = RMSE_total, 
                                PATH=Figure_path
                                )


            MAE_total_list.append(MAE_total)
            RMSE_total_list.append(RMSE_total)

            ########################################
            # Save for Total Metric of Each Metric #
            ########################################
            pd.DataFrame({
                        'MAE' : MAE,
                        'RMSE' : RMSE,
                        }, index=region_dict.values()).to_csv(f'{Pred_path}/{suptitle}_metric.csv', encoding='cp949')

            region_MAE[f'{MODEL_NAME}_{network}_{norm}_MAE'] = MAE
            region_RMSE[f'{MODEL_NAME}_{network}_{norm}_RMSE'] = RMSE
            total_metric.loc[f'{network}_{norm}',MODEL_NAME+'_MAE'] = MAE_total
            total_metric.loc[f'{network}_{norm}',MODEL_NAME+'_RMSE'] = RMSE_total
        
region_MAE.to_csv(f'{Pred_path}/region_MAE.csv', encoding='cp949')
region_RMSE.to_csv(f'{Pred_path}/region_RMSE.csv', encoding='cp949')
total_metric.to_csv(f'{Pred_path}/total_metric.csv', encoding='cp949')