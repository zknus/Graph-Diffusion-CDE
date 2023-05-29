import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function


# Define the GNN model.
class GNNhe(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNNhe, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, opt, dataset.data, device, t=time_tensor).to(device)

    if opt["use_mlp"]:
      self.reset_parameters()

    self.output_normalization =nn.LayerNorm(opt['hidden_dim'])

  def reset_parameters(self):
    torch.nn.init.xavier_normal_(self.m11.weight, gain=1.414)
    torch.nn.init.xavier_normal_(self.m12.weight, gain=1.414)

  def forward(self, x, pos_encoding=None):
    # Encode each node based on its feature.



    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper



    if self.opt['batch_norm']:
      x = self.bn_in(x)



    self.odeblock.set_x0(x)


    z = self.odeblock(x)



    # Activation.
    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)

    # Decode each node embedding to get node label.
    # z = self.output_normalization(z)

    z = self.m2(z)
    return z
