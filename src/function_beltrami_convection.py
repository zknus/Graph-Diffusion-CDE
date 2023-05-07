import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
import numpy as np
from data import get_dataset
from utils import MaxNFEException, squareplus
from base_classes import ODEFunc
import torch.nn.functional as F
from torch_scatter import scatter
import math
from torch_geometric.utils import get_laplacian

class ODEFuncBeltramiCONV(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncBeltramiCONV, self).__init__(opt, data, device)

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
    # print("self.edge_index: ", self.edge_index.shape)
    self.multihead_att_layer = SpGraphAttentionLayer(in_features, out_features,  opt,device).to(
      device)
    self.device = device

    self.edge_index, self.edge_weight = remove_self_loops(self.edge_index, self.edge_weight)

    self.edge_index_lap, self.edge_weight_lap = get_laplacian(self.edge_index, self.edge_weight, normalization='sym')
    self.edge_index_lap = self.edge_index_lap.to(device)
    self.edge_weight_lap = self.edge_weight_lap.to(device)

    self.gate = nn.Linear(2 * in_features, 1)
    nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    self.lin1 = nn.Linear(in_features, out_features)
    nn.init.xavier_normal_(self.lin1.weight, gain=1.414)

    self.lin2 = nn.Linear(in_features * 2, out_features)
    nn.init.xavier_normal_(self.lin2.weight, gain=1.414)

    self.weight_low, self.weight_high, self.weight_mlp = (
      nn.Parameter(torch.FloatTensor(in_features, out_features).to(device)),
      nn.Parameter(torch.FloatTensor(in_features, out_features).to(device)),
      nn.Parameter(torch.FloatTensor(in_features, out_features).to(device)),
    )

    self.output_low, self.output_high, self.output_mlp = (
      nn.Parameter(torch.FloatTensor(out_features, out_features).to(device)),
      nn.Parameter(torch.FloatTensor(out_features, out_features).to(device)),
      nn.Parameter(torch.FloatTensor(out_features, out_features).to(device)),
    )

    stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))

    self.weight_low.data.uniform_(-stdv, stdv)
    self.weight_high.data.uniform_(-stdv, stdv)
    self.weight_mlp.data.uniform_(-stdv, stdv)

    self.output_low.data.uniform_(-stdv, stdv)
    self.output_high.data.uniform_(-stdv, stdv)
    self.output_mlp.data.uniform_(-stdv, stdv)

    self.bn_in_1 = torch.nn.BatchNorm1d(opt['hidden_dim'])
    self.bn_in_2 = torch.nn.BatchNorm1d(opt['hidden_dim'])

  def multiply_attention(self, x, attention=None, v=None):
    num_heads = 4
    mix_features = 0
    if mix_features:
      vx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
         range(num_heads)], dim=0),
        dim=0)
      ax = self.multihead_att_layer.Wout(vx)
    else:
      mean_attention = attention.mean(dim=1)
      # mean_attention = self.edge_weight
      grad_x = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x) - x
      grad_x_abs = torch.abs(grad_x)
      grad_x_norm = torch.sqrt(torch.sum(torch.clamp(grad_x_abs * grad_x_abs, min=1e-1), 1))
      grad_x_norm_inv = 1 / grad_x_norm
      gu = grad_x_norm_inv[self.edge_index[0, :]]
      gv = grad_x_norm_inv[self.edge_index[1, :]]
      attention2 = gu * gu + gu * gv
      new_attn = mean_attention * softmax(attention2, self.edge_index[0])
      # Da = torch.diag(grad_x_norm_inv)
      W = torch.sparse.FloatTensor(self.edge_index, new_attn, (x.shape[0], x.shape[0])).coalesce()
      rowsum = torch.sparse.mm(W, torch.ones((W.shape[0], 1), device=self.device)).flatten()
      diag_index = torch.stack((torch.arange(x.shape[0]), torch.arange(x.shape[0]))).to(self.device)
      dx = torch_sparse.spmm(diag_index, rowsum, x.shape[0], x.shape[0], x)
      ax = torch_sparse.spmm(self.edge_index, new_attn, x.shape[0], x.shape[0], x)
    return ax - dx

  def forward(self, t, x):  # t is needed when called by the integrator

    attention, values = self.multihead_att_layer(x, self.edge_index)
    ax = self.multiply_attention(x, attention, values)
    # ax = self.multiply_attention(x,)
    # src = x[self.edge_index[0, :], :]
    # dst_k = x[self.edge_index[1, :], :]
    # h2 = torch.cat([src, dst_k], dim=1)
    # attention1 = torch.tanh(self.gate(h2)).squeeze()
    #
    # x_new = F.relu(torch.mm(src - dst_k, self.weight_mlp)) * dst_k
    # ax3 = scatter(x_new, self.edge_index[1, :].T, dim=0, reduce="sum")
    #
    # ax = torch.mm(ax3, self.output_high) + torch.mm(ax, self.output_low)

    # x2 = x

    src = x[self.edge_index[0, :], :]
    dst_k = x[self.edge_index[1, :], :]
    h2 = torch.cat([src, dst_k], dim=1)
    attention1 = torch.tanh(self.gate(h2)).squeeze()

    x_new = F.relu(torch.mm(src - dst_k, self.weight_mlp)) * dst_k
    # print("x_new: ", x_new.shape)

    # ax3 = torch_sparse.spmm(self.edge_index, attention1, x_new.shape[0], x_new.shape[0], x_new)

    ax3 = scatter(x_new, self.edge_index[1, :].T, dim=0, reduce="sum")

    ax3 = self.bn_in_1(ax3)
    ax = self.bn_in_2(ax)

    ax = torch.mm(ax3, self.output_high) + torch.mm(ax, self.output_low)



    ax = torch.cat([x, ax], axis=1)
    ax = self.lin2(ax)

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0

    # f = ax - x
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True):
    super(SpGraphAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = opt['heads']

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']

    self.W = nn.Parameter(torch.zeros(size=(in_features, self.attention_dim))).to(device)
    nn.init.xavier_normal_(self.W.data, gain=1.414)

    self.Wout = nn.Parameter(torch.zeros(size=(self.attention_dim, self.in_features))).to(device)
    nn.init.xavier_normal_(self.Wout.data, gain=1.414)

    self.a = nn.Parameter(torch.zeros(size=(2 * self.d_k, 1, 1))).to(device)
    nn.init.xavier_normal_(self.a.data, gain=1.414)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, x, edge):
    wx = torch.mm(x, self.W)  # h: N x out
    h = wx.view(-1, self.h, self.d_k)
    h = h.transpose(1, 2)

    # Self-attention on the nodes - Shared attention mechanism
    edge_h = torch.cat((h[edge[0, :], :, :], h[edge[1, :], :, :]), dim=1).transpose(0, 1).to(
      self.device)  # edge: 2*D x E
    edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=0)).to(self.device)
    attention = softmax(edge_e, edge[self.opt['attention_norm_idx']])
    return attention, wx

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
         'attention_norm_idx': 0, 'add_source': False,
         'alpha_dim': 'sc', 'beta_dim': 'sc', 'max_nfe': 1000, 'mix_features': False
         }
  dataset = get_dataset(opt, '../data', False)
  t = 1
  func = ODEFuncTransformerAtt(dataset.data.num_features, 6, opt, dataset.data, device)
  out = func(t, dataset.data.x)
