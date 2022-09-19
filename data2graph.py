import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import dgl
import torch
import scipy.sparse as sp

class Data2Graph:
    def __init__(self, distance_matrix, temporal_data):
        self.dist_mx = distance_matrix
        self.df = temporal_data

    def get_adjacency_matrix(self, normalized_k=0.1, int_adj = False):
        """
        Parameters
        ------------
        normalized_k
         
        int_adj
         make adjacency element to int, then adjacency matrix has not weight

        """
        # dist_mx = pd.read_csv('/Users/jeonjunhwi/문서/Projects/Master_GNN/stgcn_wave/data/sensor_graph/distances_kr_metro_city_adj_mx.csv', encoding='cp949', index_col=0)
        # self.dist_mx = self.dist_mx.values        
        
        # Calculates the standard deviation as theta.
        distances = self.dist_mx.values[~np.isinf(self.dist_mx.values)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.dist_mx.values / std))
        # Make the adjacent matrix symmetric by taking the max.
        # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

        # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
        self.adj_mx[self.adj_mx < normalized_k] = 0
        if int_adj == True:
            self.adj_mx = (self.adj_mx > 0).astype(int) # 정수일때랑 다른지 확인해보자 다르다.

        # Make Symmetric Part
        for i in range(len(self.df.columns)):
            # 연속적인 인덱스를 반환, 만약 같지 않은 부분이 있으면 건너뛰어서 불연속이 됨
            tmp = np.where(self.adj_mx[i] == self.adj_mx.transpose()[i])
            
            # 불연속 부분을 확인하기 위해 Complement를 정의
            asymset = set([i for i in range(len(self.df.columns))]) - set(tmp[0])
            for j in asymset:
                if self.adj_mx[i][j] > 0:
                    self.adj_mx[j][i] = self.adj_mx[i][j]
                    
                elif self.adj_mx[j][i] > 0:
                    self.adj_mx[i][j] = self.adj_mx[j][i]

            return self.adj_mx

    def make_corr_network(self,
                        threshold = 0.3,
                        minus_mean = False,
                        int_adj = False):#, draw=True):
        '''
        correlation이 threslohd 이상이면 연결, 아무 연결도 없는 node 들에 대해서는
        threshold보다 낮더라도 가장 큰 threshold 연결
        '''
        # correlation matrix 정의
        adj_df = self.df.corr(method='pearson')
        
        
        if minus_mean == True:
            adj_df -= adj_df.mean().mean()
            adj_df = (adj_df - adj_df.min().min()) / (adj_df.max().max() - adj_df.min().min())

        # 아래에서 max를 계산할 것이므로 대각행렬을 빼줌
        cal_1 = adj_df - np.eye(len(adj_df),len(adj_df))
        
        # threshold보다 큰 부분만 남김
        cal_2 = cal_1[cal_1 > threshold].fillna(0)

        # threshold보다 작은 부분만 남기고, 그 컬럼명 반환
        cal_3 = cal_1[cal_1 <= threshold]

        self.adj_mx = pd.DataFrame(0, index=adj_df.index, columns=adj_df.columns)
        # threshold보다 같거나 작아서 필터링된 데이터프레임에서 해당 컬럼의 가장 큰 값을 제외하고 0으로 만듦
        # print(cal_3.dropna(axis=1).columns)
        for m_c in cal_3.dropna(axis=1).columns:

            col_ = cal_3[[m_c]].sort_values(by=m_c, ascending=True)
            self.adj_mx += (self.adj_mx + col_.iloc[-1:]).fillna(0) # max value

        # threshold보다 작은 부분에서, 하나만 연결한 것과 threshold 큰 부분을 더해줌
        self.adj_mx += cal_2 + np.eye(len(adj_df),len(adj_df))
        
        if int_adj == True:
            self.adj_mx = (self.adj_mx > 0).astype(int) # 정수일때랑 다른지 확인해보자 다르다.
        # self.G = nx.Graph(self.adj_mx)
        self.adj_mx = self.adj_mx.values
        # print(self.adj_mx.type())
        
        # Make Symmetric Part
        for i in range(len(self.df.columns)):
            # 연속적인 인덱스를 반환, 만약 같지 않은 부분이 있으면 건너뛰어서 불연속이 됨
            tmp = np.where(self.adj_mx[i] == self.adj_mx.transpose()[i])
            
            # 불연속 부분을 확인하기 위해 Complement를 정의
            asymset = set([i for i in range(len(self.df.columns))]) - set(tmp[0])
            for j in asymset:
                if self.adj_mx[i][j] > 0:
                    self.adj_mx[j][i] = self.adj_mx[i][j]
                    
                elif self.adj_mx[j][i] > 0:
                    self.adj_mx[i][j] = self.adj_mx[j][i]
                    
            return self.adj_mx

    def make_dist_network(self, threshold = 3, int_adj = False):
        '''
        Make Distance Network, threshold is connection number of each node
        
        parameter
        ------------------
        threshold = int
         fix edge number of each node
        
        int_ajd = bool
         transform that adjacency matrix elements to be integer(0, 1)
         True : edge weight is 1, 0
         False : edge weight is distance value
         
        '''

        # base dataframe construct
        self.adj_mx = pd.DataFrame(0, index=self.dist_mx.index, columns=self.dist_mx.columns)

        for city in self.dist_mx.columns:
            # 오름차순 정렬
            col_ = self.dist_mx[[city]].sort_values(by=city, ascending=True)
            # 제일 작은 threshold 개만 연결. symmetric 하지 않아도 networkx에서 graph를 만들 수 있음.
            self.adj_mx += (self.adj_mx + col_.iloc[1:1+threshold]).fillna(0)
            # print(base_df)

        self.adj_mx += np.eye(len(self.adj_mx),len(self.adj_mx))
        
        if int_adj == True:
            self.adj_mx = (self.adj_mx > 0).astype(int) # 정수일때랑 다른지 확인해보자 다르다.
        self.adj_mx = self.adj_mx.values
        
        # Make Symmetric Part
        for i in range(len(self.df.columns)):
            # 연속적인 인덱스를 반환, 만약 같지 않은 부분이 있으면 건너뛰어서 불연속이 됨
            tmp = np.where(self.adj_mx[i] == self.adj_mx.transpose()[i])
            
            # 불연속 부분을 확인하기 위해 Complement를 정의
            asymset = set([i for i in range(len(self.df.columns))]) - set(tmp[0])
            for j in asymset:
                if self.adj_mx[i][j] > 0:
                    self.adj_mx[j][i] = self.adj_mx[i][j]
                    
                elif self.adj_mx[j][i] > 0:
                    self.adj_mx[i][j] = self.adj_mx[j][i]
                    
            # self.G = 
            return self.adj_mx
        
        
        
    def make_network(self, network_type, region_type, norm, int_adj):
        '''
        
        Parameters
        --------------
        network_type
         corr, dist_01, dist_02, corr-dist, dist-corr
        region_type
         state, city
        
        Returns
        --------------
         G : dgl graph
         torch.tensor(self.adj_mx, dtype=torch.float32) : Adjacency Matrix, torch.float32 format
         graph_type : str
        '''
        #########################
        ## Distance Network 01 ##
        # #######################

        if network_type == 'dist_01':
            graph_type = f'{network_type}_{region_type}'
            # dist_mx = pd.read_csv(f'data/distances_kr_{region_type}_adj_mx.csv', encoding='cp949', index_col=0)
            self.get_adjacency_matrix(normalized_k=norm, int_adj=int_adj) 
            G = dgl.from_networkx(nx.Graph(self.adj_mx))

            # self.adj_mx = torch.tensor(self.adj_mx, dtype=torch.float32)

        #########################
        ## Distance Network 02 ##
        # #######################
        
        if network_type == 'dist_02':
            graph_type = f'{network_type}_{region_type}'
            self.make_dist_network(threshold=norm, int_adj=int_adj)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))

        #########################
        ## Correlation Network ##
        #########################
        
        if network_type == 'corr':
            graph_type = f"{network_type}_{region_type}"
            self.make_corr_network(threshold=norm, minus_mean=True, int_adj=int_adj)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))   
            
        #######################
        ## Dist-Corr Network ##
        #######################
        if network_type == 'dist-corr':
            graph_type = f"{network_type}_{region_type}"
            corr_adj_mx = self.make_corr_network(threshold=0, minus_mean=True, int_adj=int_adj)
            dist_adj_mx = self.get_adjacency_matrix(normalized_k=0.1, int_adj=int_adj)

            self.adj_mx = np.where((corr_adj_mx > 0) == (dist_adj_mx > norm), corr_adj_mx, 0)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))   
            
        #######################
        ## Corr-Dist Network ##
        #######################

        if network_type == 'corr-dist':
            graph_type = f"{network_type}_{region_type}"
            corr_adj_mx = self.make_corr_network(threshold=0, minus_mean=True, int_adj=int_adj)
            dist_adj_mx = self.get_adjacency_matrix(normalized_k=0.1, int_adj=int_adj)
            
            self.adj_mx = np.where((corr_adj_mx > norm) == (dist_adj_mx > 0), dist_adj_mx, 0)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))

        ####################
        ## Complete Graph ##
        ####################

        if network_type == 'complete':
            graph_type = f"{network_type}_{region_type}"

            self.adj_mx = pd.DataFrame(1,
                                       columns=[i for i in range(len(self.df.columns))],
                                       index=[i for i in range(len(self.df.columns))])
            
            G = dgl.from_networkx(nx.Graph(self.adj_mx.values))
            
            
        # Save Network Information
        pd.DataFrame({'region' : self.dist_mx.columns,
                    'degree' : G.in_degrees()}).to_csv(f'Result/summary/{graph_type}_degree.csv', encoding='cp949')


        return G, torch.tensor(self.adj_mx, dtype=torch.float32), graph_type

    def save_graph_html(self, enc, title, save_name):
        
        
        nt3 = Network("1000px", "1500px", directed=False, heading= title, bgcolor='#222222', font_color='white')
        corrmat = pd.DataFrame(self.adj_mx, columns = [enc[i] for i in range(len(enc))], index=[enc[i] for i in range(len(enc))])
        self.G = nx.Graph(corrmat)
        # print(corrmat)
        nt3.from_nx(self.G)
        nt3.show_buttons(filter_=['physics'])
        nt3.toggle_physics(True)
        nt3.show(f"Result/html/{save_name}.html")
        
        
        # Make Heatmap
        # top_corr_features = corrmat.index
        plt.figure(figsize=(30,30))
        g=sns.heatmap(corrmat) #annot=True,

        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize=30)

        plt.yticks(fontsize=15, rotation=0)
        plt.xticks(fontsize=15, rotation=90)
        plt.title(title, fontsize=20)
        plt.savefig(f"Result/html/{save_name}_corr_heatmap.png")