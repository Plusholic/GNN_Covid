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
        
        # self loop만 존재하면 하나는 연결하는 부분 추가해야함.
        
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
            
        # dist_mx index -> df index
        self.adj_mx = self.adj_mx.reindex(self.df.columns)[self.df.columns]
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
        
    def make_cross_corr_network(self, threshold = 0.1, int_adj = False):
        # make cross correlation matrix
        # this is not symmetric matrix
        # 
        self.adj_mx = pd.DataFrame(0, index=self.df.columns, columns= self.df.columns)

        for i in range(self.df.shape[1]):
            a = self.df.iloc[:,i]
            for j in range(self.df.shape[1]):
                b = self.df.iloc[:,j]
                a = (a - np.mean(a)) / (np.std(a) * len(a))
                b = (b - np.mean(b)) / (np.std(b))
                self.adj_mx.iloc[i,j] = np.correlate(a, b)
                
        # NaN 처리
        self.adj_mx.fillna(0, inplace=True)
        for i in range(self.adj_mx.shape[1]):
            self.adj_mx.iloc[i,i] = 1
            
        self.adj_mx = self.adj_mx.round(4)
        
        self.adj_mx = self.adj_mx[self.adj_mx > threshold].fillna(0)
        
        # threshold 등 전처리 다 한다음에 남아있는 양수를 모두 1로
        if int_adj == True:
            self.adj_mx = (self.adj_mx > 0).astype(int)
        
        # Pandas DataFrame -> Numpy array
        self.adj_mx = self.adj_mx.values
        
        return self.adj_mx
        
    def make_intersect_network(self, threshold, cnt, int_adj):
        '''
        threshold : 몇 개까지의 노드를 고려해서 intersection 할 것인지
        cnt : consider range
        '''
        cross_corr_adj_mx = self.make_cross_corr_network(threshold = 0., int_adj = False)
        corr_adj_mx = self.make_corr_network(threshold = 0., int_adj=False)
        dist_adj_mx = self.get_adjacency_matrix(normalized_k = 0., int_adj=False)

        cross_corr_adj_mx = torch.tensor(cross_corr_adj_mx, dtype=torch.float32)
        corr_adj_mx = torch.tensor(corr_adj_mx, dtype=torch.float32)
        dist_adj_mx = torch.tensor(dist_adj_mx, dtype=torch.float32)

        arg_adj_corr = cross_corr_adj_mx.argsort(descending=True) # [::-1] -> 내림차순
        arg_adj_cross_corr = corr_adj_mx.argsort(descending=True)
        arg_adj_dist = dist_adj_mx.argsort(descending=True)

        new_adj_mx = pd.DataFrame(0, index=self.df.columns, columns=self.df.columns)
        
        for i, region in enumerate(self.df.columns):
            
            set1 = set(arg_adj_corr[i][0:cnt].tolist())
            set2 = set(arg_adj_cross_corr[i][0:cnt].tolist())
            set3 = set(arg_adj_dist[i][0:cnt].tolist())

            inter_list = list(set1 & set2 & set3)
            
            if len(inter_list) > threshold:
                inter_list = inter_list[:threshold] # 내림차순 정렬이니까 순서대로 10개만 뽑아줌
            
            if int_adj:
                new_adj_mx.iloc[i,inter_list] = 1
            # dist 가중치 말고는 다 0.99 이래서 별로 의미가 없음.
            elif int_adj == False:
                new_adj_mx.iloc[i,inter_list] = dist_adj_mx[i, inter_list].tolist()
            else:
                raise('adj 가중치 타입 지정해야함')

        
        # Make Symmetric Part    
        for i in range(len(self.df.columns)):
            tmp = np.where(new_adj_mx.iloc[i] == new_adj_mx.transpose().iloc[i])
            
            # 불연속 부분을 확인하기 위해 Complement를 정의
            asymset = set([i for i in range(len(self.df.columns))]) - set(tmp[0])
            for j in asymset:
                if new_adj_mx.iloc[i,j] > 0:
                    new_adj_mx.iloc[j,i] = new_adj_mx.iloc[i,j]
                    
                elif new_adj_mx.iloc[j,i] > 0:
                    new_adj_mx.iloc[i,j] = new_adj_mx.iloc[j,i]

        return new_adj_mx.values
        
    def make_network(self, network_type, region_type, norm, int_adj, Diameter_path):
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
            self.make_corr_network(threshold=norm, minus_mean=False, int_adj=int_adj)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))   
            
        #######################
        ## Dist-Corr Network ##
        #######################
        # Dist-01 로 matrix를 만들고, Correlation으로 가중치
        if network_type == 'dist-corr':
            graph_type = f"{network_type}_{region_type}"
            corr_adj_mx = self.make_corr_network(threshold=0, minus_mean=False, int_adj=int_adj)
            dist_adj_mx = self.get_adjacency_matrix(normalized_k=0.1, int_adj=False)

            self.adj_mx = np.where((corr_adj_mx > 0) == (dist_adj_mx > norm), corr_adj_mx, 0)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))   
            
        #######################
        ## Corr-Dist Network ##
        #######################
        # Correlation 으로 matrix를 만들고, dist-01로 가중치
        if network_type == 'corr-dist':
            graph_type = f"{network_type}_{region_type}"
            corr_adj_mx = self.make_corr_network(threshold=0, minus_mean=False, int_adj=False)
            dist_adj_mx = self.get_adjacency_matrix(normalized_k=0.1, int_adj=int_adj)
            
            # corr matrix가 norm보다 큰 부분은 dist_adj_mx로 가중치, 나머지는 0
            self.adj_mx = np.where((corr_adj_mx > norm) == (dist_adj_mx > 0), dist_adj_mx, 0)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))
            
        ########################
        ## Cross_Corr Network ##
        ########################

        if network_type == 'cross_corr':
            graph_type = f"{network_type}_{region_type}"
            self.make_cross_corr_network(threshold = norm, int_adj = int_adj)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))   
            
        #############################
        ## Cross_Corr-Dist Network ##
        #############################
        # Cross Correlation 으로 matrix를 만들고, dist-01로 가중치
        if network_type == 'cross_corr-dist':
            graph_type = f"{network_type}_{region_type}"
            corr_adj_mx = self.make_cross_corr_network(threshold = norm, int_adj = False)
            dist_adj_mx = self.get_adjacency_matrix(normalized_k = 0.1, int_adj=int_adj)
            
            # corr matrix가 norm보다 큰 부분은 dist_adj_mx로 가중치, 나머지는 0
            self.adj_mx = np.where((corr_adj_mx > norm) == (dist_adj_mx > 0), dist_adj_mx, 0)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))
            
        #############################
        ## Dist-Cross_Corr Network ##
        #############################
        # Dist-01 로 matrix를 만들고, Cross Correlation으로 가중치
        if network_type == 'dist-cross_corr':
            graph_type = f"{network_type}_{region_type}"
            corr_adj_mx = self.make_cross_corr_network(threshold = 0, int_adj = int_adj)
            dist_adj_mx = self.get_adjacency_matrix(normalized_k = norm, int_adj=False)

            self.adj_mx = np.where((corr_adj_mx > 0) == (dist_adj_mx > norm), corr_adj_mx, 0)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))   
        #################################
        ## dist, cross, corr intersect ##
        #################################
        
        if network_type == 'intersect':
            graph_type = f"{network_type}_{region_type}"
            self.adj_mx = self.make_intersect_network(threshold=norm[0], cnt=norm[1], int_adj=int_adj)
            G = dgl.from_networkx(nx.Graph(self.adj_mx))   
            
        ####################
        ## Complete Graph ##
        ####################

        if network_type == 'complete':
            graph_type = f"{network_type}_{region_type}"

            self.adj_mx = pd.DataFrame(1,
                                       columns=[i for i in range(len(self.df.columns))],
                                       index=[i for i in range(len(self.df.columns))]).values
            
            G = dgl.from_networkx(nx.Graph(self.adj_mx))
            
        ####################
        ## Identity Graph ##
        ####################
            
        if network_type == 'identity':
            graph_type = f"{network_type}_{region_type}"

            self.adj_mx = pd.DataFrame(0,
                                       columns=[i for i in range(len(self.df.columns))],
                                       index=[i for i in range(len(self.df.columns))])
            
            self.adj_mx = (self.adj_mx + np.eye(len(self.adj_mx),len(self.adj_mx))).values
            G = dgl.from_networkx(nx.Graph(self.adj_mx))
            
            
        # Save Network Information
        pd.DataFrame(
                    {'region' : self.df.columns,
                    'degree' : G.in_degrees()}
                     ).to_csv(f'{Diameter_path}/{graph_type}_{norm}_degree.csv', encoding='cp949')

        print('number of edges : ', self.adj_mx[self.adj_mx>0].__len__())
        print('number of nodes : ', self.adj_mx.shape[0])
        return G, torch.tensor(self.adj_mx, dtype=torch.float32), graph_type

    def save_graph_html(self, enc, title, save_name, Network_path):
        
        nt3 = Network("1000px", "1500px", directed=False, heading= title, bgcolor='#222222', font_color='white')
        # corrmat = pd.DataFrame(self.adj_mx, columns = [enc[i] for i in range(len(enc))], index=[enc[i] for i in range(len(enc))])
        tmp = pd.DataFrame(self.adj_mx, columns = self.df.columns, index=self.df.columns) # diff로 바꿨어야 함.
        
        self.G = nx.Graph(tmp)
        # print(corrmat)
        nt3.from_nx(self.G)
        nt3.show_buttons(filter_=['physics'])
        nt3.toggle_physics(True)
        nt3.show(f"{Network_path}/{save_name}.html")
        
        # Make Heatmap
        # top_corr_features = corrmat.index
        # plt.figure(figsize=(30,30))
        # g=sns.heatmap(tmp) #annot=True,

        # cbar = g.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=30)

        # plt.yticks(fontsize=15, rotation=0)
        # plt.xticks(fontsize=15, rotation=90)
        # plt.title(title, fontsize=20)
        # plt.savefig(f"Result/html/{save_name}_heatmap.png")
        # plt.savefig(f"{Network_path}/{save_name}.png")
        plt.figure(figsize=(30,30))
        ax = plt.subplot(1,1,1)
        g=sns.heatmap(tmp, ax=ax, cbar=False) #annot=True,

        # cbar = g.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=30)

        # plt.yticks(fontsize=15, rotation=0)
        ticklabel = ['' for _ in range(len(ax.get_yticklabels()))]
        ticklabel[3] = 'IC'#'Incheon'
        ticklabel[12] = "SE"#'Seoul' #6 ~ 18 Seoul
        ticklabel[26] = 'GG'#'Gyeonggi' #19 ~ 33 Gyeounggi
        ticklabel[37] = 'JB'#'Jeobuk' #34 ~ 40 전북
        ticklabel[42] = 'GJ'#'Gwangju' #41 ~ 43 광주
        ticklabel[49] = 'JN'#'Jeonnam' #44 ~ 54 전남
        ticklabel[57] = 'DG'#'Deagu' #55 ~ 58 대구
        ticklabel[65] = 'GB'#'Gyeongbuk' #59 ~ 69 경북
        ticklabel[74] = 'GN'#'Gyeongnam' #70 ~ 78 경남
        ticklabel[83] = 'BS'#'Busan' #79 ~ 86 부산
        ticklabel[88] = 'US'#'Ulsan' #87 ~ 89 울산
        ticklabel[92] = 'CB'#'Chungbuk' #90 ~ 94 충북
        ticklabel[96] = 'DJ'#'Deajeon' #95 ~ 97 대전
        ticklabel[101] = 'CN'#'Chungnam' #98 ~ 104 충남
        ticklabel[104] = 'SJ'#'Sejong' #105 ~ 105 세종 (제주랑 글씨가 겹쳐서 104로 지정)
        ticklabel[106] = 'JJ'#'Jeju' #106 ~ 106 제주
        ticklabel[111] = 'GW'#'Gangwon' #107 ~ 115 강원  
        ax.set_yticklabels(ticklabel, fontsize=50, rotation=0)
        ax.set_xticklabels(ticklabel, fontsize=50, rotation=90)
        plt.tight_layout()
        # plt.savefig(f"{Network_path}/{save_name}.png")
        plt.savefig(f"{Network_path}/{save_name}.pdf")
        
class DropEdge:
    def __init__(self, adj_mx, percent, Network_path, save_name):
        self.adj_mx = adj_mx
        self.percent = percent
        self.Network_path = Network_path
        self.save_name = save_name
        
    def dropedge(self):
        print('number of edges before drop: ', self.adj_mx[self.adj_mx>0].__len__())

        nnz =self.adj_mx.nonzero() # 연결되어있는 엣지 쌍을 반환
        perm = np.random.permutation(nnz) # 엣지 쌍의 순서를 섞어줌
        drop_nnz = int(len(nnz)*self.percent) # 엣지 쌍의 길이에 드롭 할 퍼센트를 곱해줌
        perm = perm[:drop_nnz] # 드롭할 만큼의 엣지 쌍을 반환

        row = perm.transpose(1,0)[0] # 전치 후 첫번째 열은 source node
        col = perm.transpose(1,0)[1] # 전치 후 두번째 열은 target node 
        self.adj_mx[row, col] = 0 # 이 부분을 0으로 drop 해줌

        print('number of edges after drop : ', self.adj_mx[self.adj_mx>0].__len__())
        
        # tensor의 형태로 adj_mx를 반환
        return self.adj_mx
    
    def save(self, cnt):
        tmp = pd.DataFrame(self.adj_mx.numpy())#, columns = dist_mx.columns, index = dist_mx.index)
        plt.figure(figsize=(30,30))
        g=sns.heatmap(tmp) #annot=True,

        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize=30)

        plt.yticks(fontsize=15, rotation=0)
        plt.xticks(fontsize=15, rotation=90)
        plt.savefig(f"{self.Network_path}/{self.save_name}_drop_{self.percent}_{cnt}.png")