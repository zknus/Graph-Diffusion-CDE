import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, GATConv
from torch_scatter import scatter
from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
import torch_sparse
class Lap_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, graph_size, dt=1., alpha=1., gamma=1., res_version=1,  ):
        super(Lap_GCN, self).__init__()
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
        self.graph_size = graph_size
        self.epsilons = nn.ParameterList()
        for i in range(self.nlayers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.nhid, 1))))
        # print("self.epsilons: ", self.epsilons[0].shape)
        # print("self.graph_size",self.graph_size)

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
        X = self.act_fn(self.enc(input))


        X = F.dropout(X, self.dropout, training=self.training)
        X0 =X
        for i in range(self.nlayers):

            # coeff = (1 + torch.tanh(self.epsilons[i]).tile(self.graph_size, 1))
            coeff = (1 + torch.tanh(self.epsilons[i])).T
            coeff = coeff.tile(self.graph_size, 1)
            # print("coeff shape: ", coeff.shape)
            # print("X0 shape: ", X0.shape)
            X0 = X0 * coeff  + self.dt * (self.act_fn(self.conv(X, edge_index) + self.residual(X)) - self.alpha * X)
            X = X0


            # X = X + self.dt*(self.act_fn(self.conv(X,edge_index) + self.residual(X)) - self.alpha*X)
            # X = X + self.dt * (self.act_fn(self.conv(X, edge_index)) - self.alpha * X)
            # X = X + self.dt * (self.act_fn(self.conv(X, edge_index) + self.residual(X)) )
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X


class Lap_conv_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers,graph_size, dt=1., alpha=1., gamma=1., res_version=1):
        super(Lap_conv_GCN, self).__init__()
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

        self.epsilons = nn.ParameterList()
        for i in range(self.nlayers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.nhid, 1))))
        self.lamda = nn.ParameterList()
        for i in range(self.nlayers):
            self.lamda .append(nn.Parameter(torch.zeros((self.nhid, 1))))
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
        # edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        X = self.act_fn(self.enc(input))
        self.edge_index ,self.edge_weight = add_remaining_self_loops (data.edge_index, data.edge_weight)
        edge_index = self.edge_index



        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            # X = X + self.dt*(self.act_fn(self.conv(X,edge_index) + self.residual(X)) - self.alpha*X - self.gamma*X)

            src = X[self.edge_index[0, :], :]
            dst_k = X[self.edge_index[1, :], :]
            h2 = torch.cat([src, dst_k], dim=1)
            attention1 = torch.tanh(self.gate(h2)).squeeze()

            # x_new = F.relu(torch.mm(src - dst_k, self.weight_mlp)) * dst_k
            x_new = torch.tanh(torch.mm(src - dst_k, self.weight_mlp)) * dst_k
            ax3 = scatter(x_new, self.edge_index[1, :].T, dim=0, reduce="sum")

            # ax3 = torch_sparse.spmm(self.edge_index, attention1, x_new.shape[0], x_new.shape[0], x_new)
            # ax3 = scatter(ax3, self.edge_index[1, :].T, dim=0, reduce="sum")
            ax2 = self.act_fn(self.conv(X, edge_index) + self.residual(X))

            # print("X: ", X.shape)
            # print("x_new: ", x_new.shape)
            # print("ax3: ", ax3.shape)
            # print("ax2: ", ax2.shape)

            # ax = torch.mm(ax3, self.output_high) + torch.mm(ax2, self.output_low)

            # ax = torch.cat([X, ax2], axis=1)
            # ax = self.lin2(ax)
            coeff_lamda = (torch.tanh(self.lamda[i])).T
            coeff_lamda = coeff_lamda.tile(self.graph_size, 1)

            ax = ax2 + coeff_lamda * ax3

            ax = ax - self.alpha * X

            coeff = (1 + torch.tanh(self.epsilons[i])).T
            coeff = coeff.tile(self.graph_size, 1)

            X = X * coeff + self.dt* ax

            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X