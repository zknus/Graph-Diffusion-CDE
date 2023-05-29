import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, GATConv
from torch_scatter import scatter

def batch_jacobian(func, x, create_graph=False):
  # x in shape (Batch, Length)
  def _func_sum(x):
    return func(x).sum(dim=0)

  return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 2, 0)




class attention_H(nn.Module):
  """"replace this module by a aggregation function """

  def __init__(self, size_in, edge_index):
    super().__init__()
    self.dim = size_in

    self.layer1 =GCNConv(size_in*2, size_in*2, normalize=True)
    self.edge_index = edge_index
    self.layer2 =GCNConv(size_in*2,size_in, normalize=True)

    self.layer3 = GCNConv(size_in , 1, normalize=True)
  def forward(self, x):

    out = self.layer1(x,self.edge_index)
    out = torch.tanh(out)
    out = self.layer2(out,self.edge_index)
    out = torch.tanh(out)
    out = self.layer3(out, self.edge_index)
    return out

class HAMCON_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers,data,device, dt=1., alpha=1., gamma=1., res_version=1,):
        super(HAMCON_GCN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)
        if(res_version==1):
            self.residual = self.res_connection_v1
        else:
            self.residual = self.res_connection_v2
        self.dt = dt
        self.act_fn = nn.ReLU()
        self.alpha = alpha
        self.gamma = gamma
        self.reset_params()
        self.in_features = nhid

        self.edge_index = data.edge_index.to(device)
        self.H = attention_H(self.in_features, self.edge_index)
    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

    def res_connection_v1(self, X):
        res = - self.res(self.conv.lin(X))
        return res

    def res_connection_v2(self, X):
        res = - self.conv.lin(X) + self.res(X)
        return res

    def forward(self, data):
        input = data.x
        edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            x_full = torch.hstack([X, Y])
            f_full = batch_jacobian(lambda xx: self.H(xx), x_full, create_graph=True).squeeze()
            dx = f_full[..., self.in_features:]
            dv = -1 * f_full[..., 0:self.in_features]

            # Y = Y + self.dt*( dv- self.alpha*Y - self.gamma*X)
            Y = Y + self.dt * (dv)   ###v1
            # Y = Y + self.dt * (dv - self.alpha * Y )  ##v2
            X = X + self.dt*dx
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X


class GraphCON_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, dt=1., alpha=1., gamma=1., res_version=1):
        super(GraphCON_GCN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)
        if(res_version==1):
            self.residual = self.res_connection_v1
        else:
            self.residual = self.res_connection_v2
        self.dt = dt
        self.act_fn = nn.ReLU()
        self.alpha = alpha
        self.gamma = gamma
        self.reset_params()

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

    def res_connection_v1(self, X):
        res = - self.res(self.conv.lin(X))
        return res

    def res_connection_v2(self, X):
        res = - self.conv.lin(X) + self.res(X)
        return res

    def forward(self, data):
        input = data.x
        edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            Y = Y + self.dt*(self.act_fn(self.conv(X,edge_index) + self.residual(X)) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X

class GraphCON_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, dt=1., alpha=1., gamma=1., nheads=4):
        super(GraphCON_GAT, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dropout = dropout
        self.nheads = nheads
        self.nhid = nhid
        self.nlayers = nlayers
        self.act_fn = nn.ReLU()
        self.res = nn.Linear(nhid, nheads * nhid)
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GATConv(nhid, nhid, heads=nheads)
        self.dec = nn.Linear(nhid,nclass)
        self.dt = dt

    def res_connection(self, X):
        res = self.res(X)
        return res

    def forward(self, data):
        input = data.x
        n_nodes = input.size(0)
        edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            Y = Y + self.dt*(F.elu(self.conv(X, edge_index) + self.res_connection(X)).view(n_nodes, -1, self.nheads).mean(dim=-1) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X


class GraphCON_GCN_conv(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers,graph_size, dt=1., alpha=1., gamma=1., res_version=1):
        super(GraphCON_GCN_conv, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)
        if(res_version==1):
            self.residual = self.res_connection_v1
        else:
            self.residual = self.res_connection_v2
        self.dt = dt
        self.act_fn = nn.ReLU()
        self.alpha = alpha
        self.gamma = gamma
        self.reset_params()
        self.lamda1 = nn.Parameter(torch.tensor(1.0),requires_grad=True)

        self.gate = nn.Linear(2 * nhid, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

        self.lin1 = nn.Linear(nhid, nhid)
        nn.init.xavier_normal_(self.lin1.weight, gain=1.414)

        self.lin2 = nn.Linear(nhid * 2, nhid)
        nn.init.xavier_normal_(self.lin2.weight, gain=1.414)

        self.weight_low, self.weight_high, self.weight_mlp = (
            nn.Parameter(torch.FloatTensor(nhid, nhid)),
            nn.Parameter(torch.FloatTensor(nhid, nhid)),
            nn.Parameter(torch.FloatTensor(nhid, nhid)),
        )

        self.output_low, self.output_high, self.output_mlp = (
            nn.Parameter(torch.FloatTensor(nhid, nhid)),
            nn.Parameter(torch.FloatTensor(nhid, nhid)),
            nn.Parameter(torch.FloatTensor(nhid, nhid)),
        )

        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)

        self.output_low.data.uniform_(-stdv, stdv)
        self.output_high.data.uniform_(-stdv, stdv)
        self.output_mlp.data.uniform_(-stdv, stdv)
        self.lamda = nn.ParameterList()
        for i in range(self.nlayers):
            self.lamda .append(nn.Parameter(torch.zeros((self.nhid, 1)),requires_grad=True))
        self.graph_size = graph_size

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

    def res_connection_v1(self, X):
        res = - self.res(self.conv.lin(X))
        return res

    def res_connection_v2(self, X):
        res = - self.conv.lin(X) + self.res(X)
        return res

    def forward(self, data):
        input = data.x
        edge_index = data.edge_index
        self.edge_index =  data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)



        for i in range(self.nlayers):
            coeff_lamda = (torch.tanh(self.lamda[i])).T
            coeff_lamda = coeff_lamda.tile(self.graph_size, 1)

            # src = X[self.edge_index[0, :], :]
            # dst_k = X[self.edge_index[1, :], :]
            # x_new = F.relu(torch.mm(src - dst_k, self.weight_mlp)) * dst_k
            # ax3 = scatter(x_new, self.edge_index[1, :].T, dim=0, reduce="sum")

            src = X[self.edge_index[0, :], :]
            dst_k = X[self.edge_index[1, :], :]
            x_new = torch.relu(torch.mm(src - dst_k, self.weight_mlp)) * dst_k
            ax3 = scatter(x_new, self.edge_index[1, :].T, dim=0, reduce="sum")


            Y = Y + self.dt*(self.act_fn(self.conv(X,edge_index) + self.residual(X)) - self.alpha*Y - self.gamma*X )
            X = X + self.dt* (Y + self.lamda1 * ax3)
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X
