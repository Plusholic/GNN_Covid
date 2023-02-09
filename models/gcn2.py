import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        # print(adj.shape, input.shape, h0.shape)
        # torch.Size([229, 229]) torch.Size([28, 229, 5, 2])
        # hi = torch.spmm(adj, input)
        hi = torch.einsum('ij,jklm->kilm', [adj, input.permute(1, 0, 2, 3)])
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        # print(support.shape, self.weight.shape)
        # torch.Size([28, 229, 5, 2]) torch.Size([2, 2])
        # torch.einsum('ijkl,lm->ijkm', [support, input.permute(1, 0, 2, 3)])
        output = theta*torch.matmul(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, adj_mx, TIME_STEPS, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant, FourierEmbedding):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.adj = adj_mx
        
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        
        self.FourierEmbedding = FourierEmbedding
        if self.FourierEmbedding:
            sigma = 0.2
            nhidden = 100
            self.B = nn.Parameter(
                sigma*torch.randn(1,int(0.5*nhidden),dtype=torch.float32)
                ,requires_grad=False)
        
        
        self.fcs.append(nn.Linear(nhidden*TIME_STEPS, nclass)) 
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        
    # Fourier Embedding layer funtion
    def Fourier(self,X):
        X = X@self.B
        X = torch.cat((torch.sin(X),torch.cos(X)),-1)
        return X
    
    def forward(self, x):
        
        _layers = []
        # if self.FourierEmbedding:
        #     #[B, N, T, 1]
        #     x = self.Fourier(x)
        #     #[B, N, T, 0.5*nfeat]
            
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(x)
        print(layer_inner.shape)
        _layers.append(x)
        for i,con in enumerate(self.convs):
            # layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,
                                          self.adj,
                                          _layers[0],
                                          self.lamda,
                                          self.alpha,
                                          i+1))
        print('layer_inner shape : ', layer_inner.shape)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)

        if self.FourierEmbedding:
            #[B, N, T, 16]
            x = self.Fourier(x)
            #[B, N, T, 0.5*nfeat]
        print(x.shape)
        
        layer_inner = self.fcs[-1](layer_inner.reshape(layer_inner.size(0), layer_inner.size(1), -1))
        # return F.log_softmax(layer_inner, dim=1)
        return layer_inner