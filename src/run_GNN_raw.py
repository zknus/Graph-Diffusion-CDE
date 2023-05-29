import argparse
import time
import os

import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator


from data import get_dataset, set_train_val_test_split

from best_params import best_params_dict

from utils import ROOT_DIR

import sys
import json
from GNN_heter import GNNheter
from GNN_he import GNNhe
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import is_undirected, to_undirected

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))






def train(model, optimizer, data, pos_encoding=None):
  model.train()
  optimizer.zero_grad()
  feat = data.x


  out = model(feat, pos_encoding)



  lf = torch.nn.functional.nll_loss
  loss = lf(out.log_softmax(dim=1)[data.train_mask], data.y.squeeze()[data.train_mask])






  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()





@torch.no_grad()
def test(model, data, pos_encoding=None, opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x

  logits, accs = model(feat, pos_encoding), []
  logits = F.log_softmax(logits, dim=1)
  if opt['dataset'] in [ 'minesweeper', 'workers', 'questions']:
    # print("using ROC-AUC metric")
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      # pred = logits.max(1)[1]
      # acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
      mask_idx = torch.where(mask)[0]
      y_true = data.y[mask_idx].cpu().numpy()
      y_score = logits[mask_idx].cpu().numpy()
      acc = roc_auc_score(y_true=data.y[mask_idx].cpu().numpy(),
                                         y_score=logits[:, 1][mask_idx].cpu().numpy()).item()
      accs.append(acc)

  else:

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      pred = logits[mask].max(1)[1]
      acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
      accs.append(acc)
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)



def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['attention_type'] != 'scaled_dot':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] is not None:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] != 100:
    opt['epoch'] = cmd_opt['epoch']
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']
  if cmd_opt['dropout'] is not None:
    opt['dropout'] = cmd_opt['dropout']
  if cmd_opt['hidden_dim'] is not None:
    opt['hidden_dim'] = cmd_opt['hidden_dim']
  if cmd_opt['decay'] is not None:
    opt['decay'] = cmd_opt['decay']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['edge_homo']  != 0:
    opt['edge_homo'] = cmd_opt['edge_homo']



def main(cmd_opt,split):
  # print("cmd_opt['dataset']",cmd_opt['dataset'])
  try:
    best_opt = best_params_dict[cmd_opt['dataset']]
    # print("best_opt: ", best_opt)
    opt = {**cmd_opt, **best_opt}
    # print("opt: ", opt)
    # print("opt input dropout: ", opt['input_dropout'])
    merge_cmd_args(cmd_opt, opt)
    print("merge_cmd_args from best paras")
  except KeyError:
    print("KeyError from merge_cmd_args")
    opt = cmd_opt

  dataset = get_dataset(opt, f'{ROOT_DIR}/data', True,split)
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = torch.device('cuda:' + str(opt['cuda']) if torch.cuda.is_available() else 'cpu')



  pos_encoding = None



  model = GNNhe(opt, dataset, device).to(device) if opt["no_early"] else GNNheter(opt, dataset, device).to(device)


  data = dataset.data.to(device)

  data.edge_index = to_undirected(data.edge_index)
  # print("is undirected: ", is_undirected(data.edge_index, data.edge_attr))
  # data = dataset[0].to(device)
  print("num of train samples: ", len(torch.nonzero(data.train_mask,as_tuple=True)[0]))
  print("num of val samples: ", len(torch.nonzero(data.val_mask,as_tuple=True)[0]))
  print("num of test samples: ", len(torch.nonzero(data.test_mask,as_tuple=True)[0]))

  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
  best_time = best_epoch = train_acc = val_acc = test_acc = 0

  this_test =  test

  for epoch in range(1, opt['epoch']):
    start_time = time.time()



    loss = train(model, optimizer, data, pos_encoding)
    tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, pos_encoding, opt)

    best_time = opt['time']
    if tmp_val_acc > val_acc:
      best_epoch = epoch
      train_acc = tmp_train_acc
      val_acc = tmp_val_acc
      test_acc = tmp_test_acc
      best_time = opt['time']
    if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
      best_epoch = epoch
      val_acc = model.odeblock.test_integrator.solver.best_val
      test_acc = model.odeblock.test_integrator.solver.best_test
      train_acc = model.odeblock.test_integrator.solver.best_train
      best_time = model.odeblock.test_integrator.solver.best_time

    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'

    print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, val_acc, test_acc, best_time))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc, test_acc,
                                                                                                     best_epoch,
                                                                                                     best_time))
  return train_acc, val_acc, test_acc,opt


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  parser.add_argument('--cuda', default=1, type=int)
  # data args
  parser.add_argument('--dataset', type=str, default='syn_cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv,chameleon, squirrel,'
                           'wiki-cooc, roman-empire, amazon-ratings, minesweeper, workers, questions',)
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--self_loop_weight',type=float, help='Weight of self-loops.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--geom_gcn_splits', default=True, dest='geom_gcn_splits', action='store_true',
                      help='use the 10 fixed splits from '
                           'https://arxiv.org/abs/2002.05287')
  parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                      help='the number of splits to repeat the results on')
  parser.add_argument('--label_rate', type=float, default=0.5,
                      help='% of training labels to use when --use_labels is set.')
  parser.add_argument('--planetoid_split', action='store_true',
                      help='use planetoid splits for Cora/Citeseer/Pubmed')

  parser.add_argument('--random_splits',action='store_true',help='fixed_splits')

  parser.add_argument('--edge_homo', type=float, default=0.0, help="edge_homo")


  # GNN args
  parser.add_argument('--hidden_dim', type=int,  help='Hidden dimension.')
  parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                      help='Add a fully connected layer to the decoder.')
  parser.add_argument('--input_dropout', type=float, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, help='Dropout rate.')
  parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
  parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float,  help='Weight decay for optimization')
  parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                      help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
  parser.add_argument('--block', type=str,  help='constant, mixed, attention, hard_attention')
  parser.add_argument('--function', type=str, help='laplacian, transformer, dorsey, GAT')
  parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                      help='Add a fully connected layer to the encoder.')
  parser.add_argument('--add_source', dest='add_source', action='store_true',
                      help='If try get rid of alpha param and the beta*x0 source term')
  parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

  # ODE args
  parser.add_argument('--time', type=float, help='End time of ODE integrator.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--step_size', type=float, default=1,
                      help='fixed step size when using fixed step solvers e.g. rk4')
  parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
  parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                      help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
  parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                      help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--adjoint_step_size', type=float, default=1,
                      help='fixed step size when using fixed step adjoint solvers e.g. rk4')
  parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
  parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                      help="multiplier for adjoint_atol and adjoint_rtol")
  parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
  parser.add_argument("--max_nfe", type=int, default=1000,
                      help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")
  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
  parser.add_argument("--max_test_steps", type=int, default=100,
                      help="Maximum number steps for the dopri5Early test integrator. "
                           "used if getting OOM errors at test time")

  # Attention args
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
  parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
  parser.add_argument('--attention_dim', type=int, default=64,
                      help='the size to project x to before calculating att scores')
  parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                      help='apply a feature transformation xW to the ODE')
  parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                      help="multiply attention scores by edge weights before softmax")
  parser.add_argument('--attention_type', type=str, default="scaled_dot",
                      help="scaled_dot,cosine_sim,pearson, exp_kernel")
  parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')





  args = parser.parse_args()

  opt = vars(args)

  n_splits = 10
  best = []
  train_acc_list = []
  val_acc_list = []
  timestr = time.strftime("%Y%m%d-%H%M%S")
  filename = "log/" + str(args.dataset) + str(args.function) + str(args.block) + str(args.time) + timestr + ".txt"
  command_args = " ".join(sys.argv)
  with open(filename, 'a') as f:
    json.dump(command_args, f)
    f.write("\n")

  for split in range(n_splits):
    train_acc,val_acc, test_acc, opt_final = main(opt,split)
    best.append(test_acc)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    with open(filename, 'a') as f:
      json.dump(test_acc, f)
      f.write("\n")
  print('Mean test accuracy: ', np.mean(np.array(best) * 100), 'std: ', np.std(np.array(best) * 100))
  print("test acc: ", best)

  with open(filename, 'a') as f:
    f.write(str(np.mean(np.array(best) * 100)))
    f.write(",")
    f.write(str(np.std(np.array(best) * 100)))
    f.write("\n")
    f.write("train acc list: ")
    json.dump(train_acc_list, f)
    f.write("\n")
    f.write("val acc list: ")
    json.dump(val_acc_list, f)
    f.write("\n")

    json.dump(opt_final, f, indent=2)




