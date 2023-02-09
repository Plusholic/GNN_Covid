import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GINConv
# from dgl.nn.pytorch.conv import ChebConv, GCN2Conv, GATConv


class TemporalConvLayer(nn.Module):
    ''' Temporal convolution layer.

    arguments
    ---------
    c_in : int
        The number of input channels (features)
    c_out : int
        The number of output channels (features)
    dia : int
        The dilation size
    '''
    def __init__(self, c_in, c_out, dia = 1):
        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(c_in, c_out, (2, 1), 1, dilation = dia, padding = (0,0))


    def forward(self, x): #Each sample T [1,K,207] where K is the number of steps from the past, [N,100,207]
        # print('size 확인 1 : ', x.shape)
        # print('size 확인 2 : ', self.conv(x).shape)
        return torch.relu(self.conv(x))
        
        
class SpatioConvLayer(nn.Module):
    def __init__(self, c, Lk): # c : hidden dimension Lk: graph matrix
        super(SpatioConvLayer, self).__init__()
        self.g = Lk
        self.gc = GraphConv(in_feats = c, out_feats = c)#, activation=torch.tanh)
        self.gru = nn.GRU(input_size = c, hidden_size = c, num_layers = 1, batch_first=True)
        # self.lstm = nn.LSTM(input_size = c * 17, hidden_size = c, num_layers=1)
        # self.gc2 = GraphConv(2*c, c)#, activation=torch.tanh)

 
    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # batch size, channel, n_his ,node size
        
        
        # node size, channel, n_his ,batch size
        x = x.transpose(0, 3)
        # node size, batch size, n_his ,channel
        x = x.transpose(1, 3)
        output = self.gc(self.g, x)
        # node size, batch size, n_his(conv때문에 약간 바뀜), channel
        # print(output.shape)
        
        # output, _ = self.gru(output)
        
        # input(" press ctrl + c")
        output = output.transpose(1, 3)
        output = output.transpose(0, 3)
        
        # skip connection term
        x = x.transpose(1, 3)
        x = x.transpose(0, 3)
        return torch.relu(output+x)



class FullyConvLayer(nn.Module):
    def __init__(self, c):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.fc = FullyConvLayer(c)

    def forward(self, x):
        # print(x.shape)
        x_t1 = self.tconv1(x)
        # print(x_t1.shape)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        x_t2 = x_ln
        return self.fc(x_t2)

# control_str 을 통해서 dynamic 하게 구현
class STGCN_WAVE(nn.Module):
    def __init__(self, c, T, n, Lk, p, num_layers, device, control_str = 'TNTSTNTST'): 
        super(STGCN_WAVE, self).__init__()
        self.control_str = control_str # model structure controller
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        self.BN = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)
        cnt = 0
        diapower = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == 'T': # Temporal Layer
                self.layers.append(TemporalConvLayer(c[cnt], c[cnt + 1], dia = 2**diapower))
                diapower += 1
                cnt += 1
            if i_layer == 'S': # Spatio Layer
                self.layers.append(SpatioConvLayer(c[cnt], Lk))
            if i_layer == 'N': # Norm Layer
                self.layers.append(nn.LayerNorm([n,c[cnt]]))
        self.output = OutputLayer(c[cnt], T + 1 - 2**(diapower), n) # diapower가 뭔지 모르겠네
        for layer in self.layers:
            layer = layer.to(device)
    def forward(self, x):
        x = x.permute(0,3,2,1)
        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == 'N':
                x = self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                # x = self.dropout(x)
            elif i_layer == 'T':
                x = self.layers[i](x)
                x = self.dropout(x)
            else:
                x = self.layers[i](x)
        
        out = self.output(x) # [batch, 1, 1, node]
        # print(out.shape)
        return out.reshape(out.size(0), out.size(3), 1)


