import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import dgl
import torch



def get_adjacency_matrix(dist_mx, normalized_k=0.1, int_adj = False):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return: adjacency matrix
    """
    # dist_mx = pd.read_csv('/Users/jeonjunhwi/문서/Projects/Master_GNN/stgcn_wave/data/sensor_graph/distances_kr_metro_city_adj_mx.csv', encoding='cp949', index_col=0)
    dist_mx = dist_mx.values        
        
    # print(dist_mx)
    # input("press_enter : ")
    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    if int_adj == True:
        adj_mx = (adj_mx > 0).astype(int) # 정수일때랑 다른지 확인해보자 다르다.
    return adj_mx


def make_corr_network(data,
                      threshold = 0.3,
                      minus_mean = False,
                      int_adj = False):#, draw=True):
    '''
    correlation이 threslohd 이상이면 연결, 아무 연결도 없는 node 들에 대해서는
    threshold보다 낮더라도 가장 큰 threshold 연결
    '''
    # correlation matrix 정의
    adj_df = data.corr(method='pearson')
    
    
    if minus_mean == True:
        adj_df -= adj_df.mean().mean()
        adj_df = (adj_df - adj_df.min().min()) / (adj_df.max().max() - adj_df.min().min())

    # 아래에서 max를 계산할 것이므로 대각행렬을 빼줌
    cal_1 = adj_df - np.eye(len(adj_df),len(adj_df))
    
    # threshold보다 큰 부분만 남김
    cal_2 = cal_1[cal_1 > threshold].fillna(0)

    # threshold보다 작은 부분만 남기고, 그 컬럼명 반환
    cal_3 = cal_1[cal_1 <= threshold]

    adj_mx = pd.DataFrame(0, index=adj_df.index, columns=adj_df.columns)
    # threshold보다 같거나 작아서 필터링된 데이터프레임에서 해당 컬럼의 가장 큰 값을 제외하고 0으로 만듦
    # print(cal_3.dropna(axis=1).columns)
    for m_c in cal_3.dropna(axis=1).columns:

        col_ = cal_3[[m_c]].sort_values(by=m_c, ascending=True)
        adj_mx += (adj_mx + col_.iloc[-1:]).fillna(0) # max value

    # threshold보다 작은 부분에서, 하나만 연결한 것과 threshold 큰 부분을 더해줌
    adj_mx += cal_2 + np.eye(len(adj_df),len(adj_df))
    
    if int_adj == True:
        adj_mx = (adj_mx > 0).astype(int) # 정수일때랑 다른지 확인해보자 다르다.
    G = nx.Graph(adj_mx)
    
    return G, adj_mx


def make_dist_network(dist_mx, threshold = 3, int_adj = False):#, draw=True):
    '''
    data is distance network
    '''

    # base dataframe construct
    base_df = pd.DataFrame(0, index=dist_mx.index, columns=dist_mx.columns)

    for city in dist_mx.columns:
        # 오름차순 정렬
        col_ = dist_mx[[city]].sort_values(by=city, ascending=True)
        # 제일 작은 threshold 개만 연결. symmetric 하지 않아도 networkx에서 graph를 만들 수 있음.
        base_df += (base_df + col_.iloc[1:1+threshold]).fillna(0)
        # print(base_df)

    adj_mx += np.eye(len(base_df),len(base_df))
    G = nx.Graph(adj_mx)
    
    if int_adj == True:
        adj_mx = (adj_mx > 0).astype(int) # 정수일때랑 다른지 확인해보자 다르다.
        
    return G, adj_mx


def save_graph_html(adj_mx, enc, title, save_name):
    
    
    nt3 = Network("1000px", "1500px", directed=False, heading= title, bgcolor='#222222', font_color='white')
    corrmat = pd.DataFrame(adj_mx, columns = [enc[i] for i in range(len(enc))], index=[enc[i] for i in range(len(enc))])
    G = nx.Graph(corrmat)
    # print(corrmat)
    nt3.from_nx(G)
    nt3.show_buttons(filter_=['physics'])
    nt3.toggle_physics(True)
    nt3.show(f"{save_name}.html")
    
    
    # Make Heatmap
    # top_corr_features = corrmat.index
    plt.figure(figsize=(30,30))
    g=sns.heatmap(corrmat) #annot=True,

    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)

    plt.yticks(fontsize=15, rotation=0)
    plt.xticks(fontsize=15, rotation=90)
    plt.title(title, fontsize=20)
    plt.savefig(f"{save_name}_corr_heatmap.png")
    
    
    
# def make_network(region_type, make_int):
#     #########################
#     ## Distance Network 01 ##
#     # #######################
#     # norm = 0.9
#     # graph_type = f'dist_01_{region_type}'
#     # dist_mx = pd.read_csv(f'data/distances_kr_{region_type}_adj_mx.csv', encoding='cp949', index_col=0)
#     # adj_mx = get_adjacency_matrix(dist_mx, normalized_k=norm) #Generates the adjacent matrix from the distance between sensors and the sensor ids. 
#     # sp_mx = sp.coo_matrix(adj_mx)
#     # G = dgl.from_scipy(sp_mx)
#     # adj_mx = torch.tensor(adj_mx, dtype=torch.float32)
#     # pd.DataFrame({'region' : dist_mx.columns, 'degree' : G.in_degrees()}).to_csv(f'Result/summary/{graph_type}_degree.csv', encoding='cp949')

#     #########################
#     ## Distance Network 02 ##
#     # #######################
#     # graph_type = f'dist_02_{region_type}'
#     # # distance_df = pd.read_csv('/Users/jeonjunhwi/문서/Projects/Master_GNN//stgcn_wave/data/sensor_graph/distances_kr_metro_city.csv', dtype={'from': 'str', 'to': 'str'}, index_col=0)
#     # dist_mx = pd.read_csv(f'data/distances_kr_{region_type}_adj_mx.csv', encoding='cp949', index_col=0)
#     # norm = 3
#     # G, adj_mx = make_dist_network(dist_mx, threshold=norm)
#     # sp_mx = sp.coo_matrix(adj_mx)
#     # G = dgl.from_scipy(sp_mx)
#     # adj_mx = torch.tensor(adj_mx.values, dtype=torch.float32)

#     #########################
#     ## Correlation Network ##
#     #########################
#     norm = 0.3
#     graph_type = f"Corr_{region_type}"
#     # daily_df_state = pd.read_csv('/Users/jeonjunhwi/문서/Projects/Master_GNN/stgcn_wave/data/smoothing_3_state_mean.csv', encoding="euc-kr",index_col =0)
#     G, adj_mx = make_corr_network(df, threshold=norm, minus_mean=True)
#     G = dgl.from_networkx(G)
#     adj_mx = (adj_mx > 0).astype(int) # 정수일때랑 다른지 확인해보자 다르다.
#     # adj_mx = torch.tensor(adj_mx.values, dtype=torch.float32)
    
#     return G, torch.tensor(adj_mx.values, dtype=torch.float32), graph_type
