import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from lib import DropEdge

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj_mx,
                 bias=False, norm=True, activation=None,
                 ):
        super(GraphConv, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.adj_mx = adj_mx
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.activation = activation
        self.norm = norm
        # DEFINE MEAN AGGREGATE
        if norm:
            # self.norm = torch.pow(node_size, -0.5) # 이거 어떻게 해야하지..? node size가 아니라 매트릭스가 와야함.
            self.norm = [len(self.adj_mx[i][self.adj_mx[i]>0]) for i in range(self.adj_mx.shape[0])]
            self.norm = torch.tensor(self.norm, dtype=torch.float32)
            self.norm = torch.pow(self.norm, -0.5)
            
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        # print(self.bias)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, features):#, adj_mx):
        if self.norm is not False:
            features = torch.einsum('j,jklm->kjlm', [self.norm, features.permute(1, 0, 2, 3)])
        else:
            features = torch.einsum('ij,jklm->kilm', [self.adj_mx, features.permute(1, 0, 2, 3)])
        # [B, N, T, in_feat] * [in_feat, out_feat]
        output = torch.matmul(features, self.weights)
        # [B, N, T, out_feat] = [N, N] * [N, B, T, out_feat]
        # output = torch.einsum('ij,jklm->kilm', [self.adj_mx, output.permute(1, 0, 2, 3)])
        if self.bias is not None:
            output += self.bias
            
        if self.activation is not None:
            output = self.activation(output)
        return output

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 adj_mx, activation, dropout, residual, batchnorm, gnn_norm, FourierEmbedding,
                 device):
        super(GCNLayer, self).__init__()
        
        self.activation = activation
            
        self.graph_conv = GraphConv(in_features=in_features,
                                    out_features=out_features,
                                    adj_mx = adj_mx,
                                    bias=False,
                                    norm=gnn_norm,
                                    activation=None
                                    )
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_features, out_features)
        
        self.bn = batchnorm
        if batchnorm:
            # self.bn_layer=nn.BatchNorm1d(out_features)
            self.bn_layer = nn.BatchNorm2d(adj_mx.shape[0]) # node size로 batch normalize 하면 히든 피처가 나눠지는건가?
        self.FourierEmbedding = FourierEmbedding
        if self.FourierEmbedding:
            sigma = 0.02
            # sigma = 0.2
            # out_features = out_features * 2
            self.B = nn.Parameter(
                sigma*torch.randn(in_features,int(0.5*out_features),dtype=torch.float32)
                ,requires_grad=False)

        
    # Fourier Embedding layer funtion
    def Fourier(self,X):
        X = X@self.B
        X = torch.cat((torch.sin(X),torch.cos(X)),-1)
        return X
            
    def reset_parameter(self):
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()
            
    def forward(self, feats):
            
        # [B, N, T, feat]
        new_feats = self.graph_conv(feats)
        if self.residual:
            # [B, N, T, feat]
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        # [B, N, T, feat]
        new_feats = self.dropout(new_feats)
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        # print('GL output', new_feats)
        
        if self.FourierEmbedding:
            #[B, N, T, feat]
            feats = self.Fourier(feats)
            #[B, N, T, 0.5*nfeat]
        
        return new_feats
    
class GCN(nn.Module):

    def __init__(self, in_feats, adj_mx, TIME_STEPS, hidden_feats=None, activation=None,
                 residual=None, batchnorm=None, dropout=None, gnn_norm=None, FourierEmbedding=None,
                 device=None,
                 ):
        super(GCN, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64]

        n_layers = len(hidden_feats)
        if gnn_norm is None:
            gnn_norm = [True for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(activation),
                   len(residual), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, gnn_norm, ' \
                                       'activation, residual, batchnorm and dropout to ' \
                                       'be the same, got {}'.format(lengths)
        self.device = device
        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayer(in_features=in_feats,
                                            out_features=hidden_feats[i],
                                            adj_mx = adj_mx,
                                            activation=activation[i],
                                            residual=residual[i],
                                            batchnorm=batchnorm[i],
                                            dropout=dropout[i],
                                            gnn_norm=gnn_norm[i],
                                            FourierEmbedding=FourierEmbedding,
                                            device=self.device,
                                            ))
            in_feats = hidden_feats[i]
        self.fc = nn.Linear(hidden_feats[-1] * TIME_STEPS,1)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()


    def forward(self, feats):#, adj_mx):

        for gnn in self.gnn_layers:
            feats = gnn(feats)#, adj_mx)
        feats = self.fc(feats.reshape(feats.size(0), feats.size(1), -1))
        # print('GCN output', feats)
        
        return feats
