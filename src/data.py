"""
Code partially copied from 'Diffusion Improves Graph Learning' repo https://github.com/klicperajo/gdc/blob/master/data.py
"""

import os

import numpy as np
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor,Airports
from torch_geometric.data import Dataset
from graph_rewiring import get_two_hop, apply_gdc
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, dense_to_sparse,from_scipy_sparse_matrix, degree
from graph_rewiring import make_symmetric, apply_pos_dist_rewire
from heterophilic import  Actor, get_fixed_splits, generate_random_splits,Planetoid2, random_disassortative_splits,CustomDataset_cora
from heterophilic import WebKB, WikipediaNetwork, Actor
from utils import ROOT_DIR
import os.path as osp
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import degree,homophily
# from torch_geometric.datasets import WebKB, WikipediaNetwork
import sys
# sys.path.append('/home/ntu/Documents/zk/ACM-GNN/synthetic-experiments/')
# sys.path.insert(0,'/home/ntu/Documents/zk/ACM-GNN/synthetic-experiments/')

# import os
# path = '/home/ntu/Documents/zk/ACM-GNN/synthetic-experiments/'
# os.environ['PATH'] += ':'+path
class MyOwnDataset(InMemoryDataset):
  def __init__(self, root, name, transform=None, pre_transform=None):
    super().__init__(None, transform, pre_transform)

def bin_feat(feat, bins):
  digitized = np.digitize(feat, bins)
  return digitized - digitized.min()

def load_data_airport(dataset_str, data_path, return_label=True):
  graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
  adj = nx.adjacency_matrix(graph)
  features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
  if return_label:
    label_idx = 4
    labels = features[:, label_idx]
    features = features[:, :label_idx]
    labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
    return sp.csr_matrix(adj), features, labels
  else:
    return sp.csr_matrix(adj), features


DATA_PATH = f'{ROOT_DIR}/data'


def rewire(data, opt, data_dir):
  rw = opt['rewiring']
  if rw == 'two_hop':
    data = get_two_hop(data)
  elif rw == 'gdc':
    data = apply_gdc(data, opt)
  elif rw == 'pos_enc_knn':
    data = apply_pos_dist_rewire(data, opt, data_dir)
  return data

def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]

    print("number of total positive samples: ", len(pos_idx))
    print("number of total negetive samples: ", len(neg_idx))
    print("number of training positive samples: ", len(idx_train_pos))
    print("number of training negetive samples: ", len(idx_train_neg))
    print("number of val positive samples: ", len(idx_val_pos))
    print("number of val negetive samples: ", len(idx_val_neg))
    print("number of val positive samples: ", len(idx_test_pos))
    print("number of val negetive samples: ", len(idx_test_neg))

    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def load_synthetic_data(dataset_str, use_feats, data_path):
  object_to_idx = {}
  idx_counter = 0
  edges = []
  with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
    all_edges = f.readlines()
  for line in all_edges:
    n1, n2 = line.rstrip().split(',')
    if n1 in object_to_idx:
      i = object_to_idx[n1]
    else:
      i = idx_counter
      object_to_idx[n1] = i
      idx_counter += 1
    if n2 in object_to_idx:
      j = object_to_idx[n2]
    else:
      j = idx_counter
      object_to_idx[n2] = j
      idx_counter += 1
    edges.append((i, j))
  adj = np.zeros((len(object_to_idx), len(object_to_idx)))
  for i, j in edges:
    adj[i, j] = 1.  # comment this line for directed adjacency matrix
    adj[j, i] = 1.
  if use_feats:
    features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
  else:
    features = sp.eye(adj.shape[0])
  labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
  return sp.csr_matrix(adj), features, labels
def get_dataset(opt: dict, data_dir, use_lcc: bool = False, split=0) -> InMemoryDataset:
  ds = opt['dataset']
  path = os.path.join(data_dir, ds)
  if ds in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(path, ds)
    use_lcc = False
    if opt["random_splits"]:
      data = generate_random_splits(dataset.data, train_rate=0.6, val_rate=0.2)
      dataset.data = data
      print("random_splits with train_rate=0.6, val_rate=0.2")
  elif ds in ['cora_lp','citeseer_lp','pubmed_lp','disease_lp','airport_lp']:

    if ds == 'cora_lp':
      ds = 'cora'
      data_path = os.path.join('/home/ntu/Documents/zk/GeoGNN/data', ds)
      adj, features = load_citation_data(ds, use_feats=1, data_path=data_path)[:2]
    if ds == 'citeseer_lp':
      ds = 'citeseer'
      data_path = os.path.join('/home/ntu/Documents/zk/GeoGNN/data', ds)
      adj, features = load_citation_data(ds, use_feats=1, data_path=data_path)[:2]
    if ds == 'pubmed_lp':
      ds = 'pubmed'
      data_path = os.path.join('/home/ntu/Documents/zk/GeoGNN/data', ds)
      adj, features = load_citation_data(ds, use_feats=1, data_path=data_path)[:2]
    if ds == 'airport_lp':
      ds = 'airport'
      data_path = os.path.join('/home/ntu/Documents/zk/GeoGNN/data', ds)
      adj, features = load_data_airport(ds, data_path, return_label=False)
    if ds == 'disease_lp':
      data_path = os.path.join('/home/ntu/Documents/zk/GeoGNN/data', ds)
      adj, features = load_synthetic_data(ds, 1, data_path)[:2]
    data = {'adj_train': adj, 'features': features}

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
      adj, 0.05, 0.1, split
    )


    data['adj_train_norm'], data['features'] = process(
      data['adj_train'], data['features'], normalize_adj = 1, normalize_feats = 1)
    if ds == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])

    dataset = MyOwnDataset(path, name=ds)
    edge_index, edge_attr = from_scipy_sparse_matrix(data['adj_train'])
    edge_attr = torch.tensor(edge_attr,dtype=torch.float32)
    data1 = Data(
      x=data['features'],
      edge_index=torch.LongTensor(edge_index),
      edge_attr=edge_attr,
    )

    data1['adj_train'] = adj_train
    data1['train_edges'], data1['train_edges_false'] = train_edges, train_edges_false
    data1['val_edges'], data1['val_edges_false'] = val_edges, val_edges_false
    data1['test_edges'], data1['test_edges_false'] = test_edges, test_edges_false

    dataset.data = data1
    use_lcc =False






  elif ds in ['Computers', 'Photo']:
    dataset = Amazon(path, ds)
  elif ds == 'CoauthorCS':
    dataset = Coauthor(path, 'CS')
  elif ds in ['cornell', 'texas', 'wisconsin','cora_acm','citeseer_acm','pubmed_acm']:
    # dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
    # dataset = WebKB(path, name=ds)
    dataset = MyOwnDataset(path, name=ds)
    adj, features, labels = load_full_data(ds)

    edge_index,edge_attr =  from_scipy_sparse_matrix(adj)
    edge_attr = torch.tensor(edge_attr,dtype=torch.float32)
    # train_mask, val_mask, test_mask = random_disassortative_splits(labels, labels.max() + 1)
    # degree_index = degree(edge_index)




    data = Data(
      x=features,
      edge_index=torch.LongTensor(edge_index),
      edge_attr=edge_attr,
      y=labels,
    )
    dataset.data = data

    # labels = dataset.data.y
    # degree_index = degree(data.edge_index[0], data.num_nodes)
    # dvk = 0
    # for i in range(0, torch.max(labels) + 1):
    #   dv = 0
    #   for j in range(len(labels)):
    #
    #     if labels[j] == i:
    #       dv = dv + degree_index[j]
    #   dvk = dvk + dv * dv
    #
    # print("DK 2: ", dvk)
    # edge_homo = homophily(data.edge_index, data.y, method='edge')
    # print("edge_homo: ", edge_homo)
    # num_edge = data.num_edges
    # print("num_edge: ", num_edge)
    # print("(dvk / (4 * num_edge * num_edge)): ",(dvk / (4 * num_edge * num_edge)))
    # ad_homo = (edge_homo - (dvk / (4 * num_edge * num_edge))) / (1.0 - (dvk / (4 * num_edge * num_edge)))
    # print("adjusted homo: ", ad_homo)


    if opt["random_splits"]:
      data = generate_random_splits(dataset.data, train_rate=0.6, val_rate=0.2)
      dataset.data = data
      print("random_splits with train_rate=0.6, val_rate=0.2")
    else:
      data = get_fixed_splits(dataset.data, ds, path, split)
      dataset.data = data
      print("fixed_splits with splits number: ", split)
    use_lcc = False

  elif ds in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(root=path, name=ds, transform=T.NormalizeFeatures())
    if opt["random_splits"]:
      data = generate_random_splits(dataset.data, train_rate=0.6, val_rate=0.2)
      dataset.data = data
      print("random_splits with train_rate=0.6, val_rate=0.2")
    else:
      data = get_fixed_splits(dataset.data, ds, path, split)
      dataset.data = data
      print("fixed_splits with splits number: ", split)
    # labels = dataset.data.y
    # degree_index = degree(data.edge_index[0], data.num_nodes)
    # dvk = 0
    # for i in range(0, torch.max(labels) + 1):
    #   dv = 0
    #   for j in range(len(labels)):
    #
    #     if labels[j] == i:
    #       dv = dv + degree_index[j]
    #   dvk = dvk + dv * dv
    #
    # print("DK 2: ", dvk)
    # edge_homo = homophily(data.edge_index, data.y, method='edge')
    # print("edge_homo: ", edge_homo)
    # num_edge = data.num_edges
    # print("num_edge: ", num_edge)
    # ad_homo = (edge_homo - (dvk / (4 * num_edge * num_edge))) / (1.0 - (dvk / (4 * num_edge * num_edge)))
    # print("adjusted homo: ", ad_homo)


    use_lcc = False
  elif ds == 'film':
    dataset = Actor(root=path, transform=T.NormalizeFeatures())
    if opt["random_splits"]:
      data = generate_random_splits(dataset.data, train_rate=0.6, val_rate=0.2)
      dataset.data = data
      print("random_splits with train_rate=0.6, val_rate=0.2")
    else:
      data = get_fixed_splits(dataset.data, ds, path, split)
      dataset.data = data
      print("fixed_splits with splits number: ", split)
    use_lcc = False
  elif ds in ['wiki-cooc', 'roman-empire', 'amazon-ratings', 'minesweeper', 'workers', 'questions']:
    dataset = MyOwnDataset(path, name=ds)
    data = np.load(os.path.join('./HeterophilousDatasets/data', f'{ds.replace("-", "_")}.npz'))
    node_features = torch.tensor(data['node_features'])
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges'])
    edges = edges.T

    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])

    # train_idx_list = [torch.where(train_mask)[0] for train_mask in train_mask]
    # val_idx_list = [torch.where(val_mask)[0] for val_mask in val_mask]
    # test_idx_list = [torch.where(test_mask)[0] for test_mask in test_mask]
    train_mask = train_masks[split,:]
    val_mask = val_masks[split, :]
    test_mask = test_masks[split, :]

    print("fixed_splits with splits number: ", split)

    data = Data(
      x=node_features,
      edge_index=torch.LongTensor(edges),
      y=labels,
      train_mask=train_mask,
      test_mask=test_mask,
      val_mask=val_mask
    )

    use_lcc = False
    dataset.data = data

    # degree_index = degree(data.edge_index[0], data.num_nodes)
    # dvk = 0
    # for i in range(0, torch.max(labels) + 1):
    #   dv = 0
    #   for j in range(len(labels)):
    #
    #     if labels[j] == i:
    #       dv = dv + degree_index[j]
    #   dvk = dvk + dv * dv
    #
    # print("DK 2: ", dvk )
    # edge_homo = homophily(data.edge_index, data.y, method='edge')
    # print("edge_homo: ",edge_homo)
    # num_edge = data.num_edges
    # print("num_edge: ", num_edge)
    # print("(dvk / (4 * num_edge * num_edge)): ", (dvk / (4 * num_edge * num_edge)))
    # ad_homo = (edge_homo - (dvk/(4 * num_edge *num_edge))) / (1.0 - (dvk/(4 * num_edge *num_edge)))
    # print("adjusted homo: ", ad_homo)



    y_train = data.y[train_mask]
    y_test = data.y[test_mask]
    y_val = data.y[val_mask]
    indices_train = []
    num_classes = len(torch.unique(data.y))
    for i in range(num_classes):
      index = (y_train == i).nonzero().view(-1)
      index = index[torch.randperm(index.size(0))]
      indices_train.append(len(index))
    print("label distribution of train: ", indices_train)

    indices_test = []

    for i in range(num_classes):
      index = (y_test == i).nonzero().view(-1)
      index = index[torch.randperm(index.size(0))]
      indices_test.append(len(index))
    print("label distribution of test: ", indices_test)

    indices_val = []

    for i in range(num_classes):
      index = (y_val == i).nonzero().view(-1)
      index = index[torch.randperm(index.size(0))]
      indices_val.append(len(index))
    print("label distribution of val: ", indices_val)

  elif ds in ['cora_gene','citeseer_gene','pubmed_gene']:
    # name = "GenCAT_cora_8_0"
    # dataset = Planetoid2("/home/ntu/Documents/zk/empirical-study-of-GNNs/data/GenCAT_Exp_hetero_homo/"+name, name, transform=T.NormalizeFeatures())
    # use_lcc = False
    #
    # if opt["random_splits"]:
    #   data = generate_random_splits(dataset.data, train_rate=0.6, val_rate=0.2,Flag=0)
    #
    #   dataset.data = data
    #   print("random_splits with train_rate=0.6, val_rate=0.2")
    print("edge_homo: ", opt['edge_homo'])
    ds_name = ds
    if ds == 'cora_gene':
      ds = 'cora'
    if ds == 'citeseer_gene':
      ds = 'citeseer'
    if ds == 'pubmed_gene':
      ds = 'pubmed'

    if opt['random_splits']:
      adj, labels, _, features = load_synthetic_data_heter(
        'random', split, opt['edge_homo'], ds
      )
    else:
      adj, labels, _, features = load_synthetic_data_heter(
        'regular', split, opt['edge_homo'], ds
      )

    dataset = MyOwnDataset(path, name=ds_name)
    adj_sparse = dense_to_sparse(adj)
    # data = dataset[0]
    edge_index = adj_sparse[0]
    edge_attr = adj_sparse[1]

    train_mask, val_mask, test_mask = random_disassortative_splits(labels, labels.max() + 1)

    data = Data(
      x=features,
      edge_index=torch.LongTensor(edge_index),
      edge_attr=edge_attr,
      y=labels,
      train_mask=train_mask,
      test_mask=test_mask,
      val_mask=val_mask
    )

    use_lcc = False
    dataset.data = data
  # elif ds == 'cora_gene':
  #   dataset = CustomDataset_cora(root=path, name="h0.00-r1", setting="gcn", seed=15)
  #
  #   adj = dataset.adj  # Access adjacency matrix
  #   features = dataset.features  # Access node features

  elif ds == 'syn_cora':
    print("edge_homo: ", opt['edge_homo'])
    from torch_geometric.io import read_npz

    path_cora = '/home/ntu/Documents/zk/graph-neural-pde/src/archives/syn-cora/h'+str(opt['edge_homo']) + '0-r' + str(split) +'.npz'

    dataset = MyOwnDataset(path_cora, name=ds)
    data= read_npz(path_cora)
    data = generate_random_splits(data, train_rate=0.25, val_rate=0.25)
    # data.edge_index = torch.LongTensor(data.edge_index),

    dataset.data = data
    use_lcc = False

    # from deeprobust.graph.data.dataset import CustomDataset

    # Load the dataset in file `syn-cora/h0.00-r1.npz`
    # `seed` controls the generation of training, validation and test splits
    # dataset = CustomDataset(root="syn-cora", name="h0.00-r1", setting="gcn", seed=15)
    #
    # adj = dataset.adj  # Access adjacency matrix
    # features = dataset.features  # Access node features


  elif ds == 'ogbn-arxiv':
    dataset = PygNodePropPredDataset(name=ds, root=path,
                                     transform=T.ToSparseTensor())
    use_lcc = False  # never need to calculate the lcc with ogb datasets
  elif ds == 'airport':
    dataset = MyOwnDataset(path, name=ds)
    adj, features,labels = load_data_airport('airport', os.path.join('../dataset', 'airport'), return_label=True)

    val_prop, test_prop = 0.15, 0.15
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=1234)
    train_mask = torch.zeros(features.shape[0], dtype=bool, )
    train_mask[idx_train] = True
    test_mask = torch.zeros(features.shape[0], dtype=bool, )
    test_mask[idx_val] = True
    val_mask = torch.zeros(features.shape[0], dtype=bool, )
    val_mask[idx_test] = True
    adj =adj.tocoo()
    row, col,edge_attr = adj.row,adj.col,adj.data
    row =torch.LongTensor(row)
    col = torch.LongTensor(col)
    edge_attr = torch.FloatTensor(edge_attr)
    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features)
    edges = torch.stack([row, col], dim=0)
    data = Data(
      x=features,
      edge_index=torch.LongTensor(edges),
      edge_attr=edge_attr,
      y=labels,
      train_mask=train_mask,
      test_mask=test_mask,
      val_mask=val_mask
    )
    use_lcc = False
    dataset.data = data

  elif ds == 'disease':
    dataset = Planetoid(path, 'cora')
    adj, features, labels = load_synthetic_data('disease_nc', 1,os.path.join('../dataset', 'disease_nc'), )
    val_prop, test_prop = 0.10, 0.60

    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=1234)
    train_mask = torch.zeros(features.shape[0], dtype=bool, )
    train_mask[idx_train] = True
    test_mask = torch.zeros(features.shape[0], dtype=bool, )
    test_mask[idx_val] = True
    val_mask = torch.zeros(features.shape[0], dtype=bool, )
    val_mask[idx_test] = True
    adj = adj.tocoo()
    row, col, edge_attr = adj.row, adj.col, adj.data
    row = torch.LongTensor(row)
    col = torch.LongTensor(col)
    edge_attr = torch.FloatTensor(edge_attr)
    labels = torch.LongTensor(labels)
    features = features.toarray()
    features = torch.FloatTensor(features)
    edges = torch.stack([row, col], dim=0)
    data = Data(
      x=features,
      edge_index=torch.LongTensor(edges),
      edge_attr=edge_attr,
      y=labels,
      train_mask=train_mask,
      test_mask=test_mask,
      val_mask=val_mask
    )
    use_lcc = False
    dataset.data = data
  else:
    raise Exception('Unknown dataset.')

  if use_lcc:
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
      x=x_new,
      edge_index=torch.LongTensor(edges),
      y=y_new,
      train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data
  if opt['rewiring'] is not None:
    dataset.data = rewire(dataset.data, opt, data_dir)
  train_mask_exists = True
  try:
    dataset.data.train_mask
  except AttributeError:
    train_mask_exists = False

  if ds == 'ogbn-arxiv':
    split_idx = dataset.get_idx_split()
    ei = to_undirected(dataset.data.edge_index)
    data = Data(
    x=dataset.data.x,
    edge_index=ei,
    y=dataset.data.y,
    train_mask=split_idx['train'],
    test_mask=split_idx['test'],
    val_mask=split_idx['valid'])
    dataset.data = data
    train_mask_exists = True

  #todo this currently breaks with heterophilic datasets if you don't pass --geom_gcn_splits
  if (use_lcc or not train_mask_exists) and not opt['geom_gcn_splits']:
    dataset.data = set_train_val_test_split(
      12345,
      dataset.data,
      num_development=5000 if ds == "CoauthorCS" else 1500)


  return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
  rnd_state = np.random.RandomState(seed)
  num_nodes = data.y.shape[0]
  development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
  test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

  train_idx = []
  rnd_state = np.random.RandomState(seed)
  for c in range(data.y.max() + 1):
    class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

  val_idx = [i for i in development_idx if i not in train_idx]

  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

  data.train_mask = get_mask(train_idx)
  data.val_mask = get_mask(val_idx)
  data.test_mask = get_mask(test_idx)

  return data

def load_synthetic_data_heter(graph_type, graph_idx, edge_homo, feature_base_name):
    Path(f"./synthetic_graphs/{graph_type}/{feature_base_name}/{edge_homo}/").mkdir(
      parents=True, exist_ok=True
    )
    adj = (
      torch.load(
        (
          f"./synthetic_graphs/{graph_type}/{feature_base_name}/{edge_homo}/adj_{edge_homo}_{graph_idx}.pt"
        )
      ).to_dense().clone().detach().float()
    )
    labels = (
      (
        np.argmax(
          torch.load(
            (
              f"./synthetic_graphs/{graph_type}/{feature_base_name}/{edge_homo}/label_{edge_homo}_{graph_idx}.pt"
            )
          )
            .to_dense()
            .clone()
            .detach()
            .float(),
          axis=1,
        )
      ).clone().detach()
    )
    degree = (
      torch.load(
        (
          f"./synthetic_graphs/{graph_type}/{feature_base_name}/{edge_homo}/degree_{edge_homo}_{graph_idx}.pt"
        )
      ).to_dense().clone().detach().float()
    )

    if feature_base_name in {
      "CitationFull_dblp",
      "Coauthor_CS",
      "Coauthor_Physics",
      "Amazon_Computers",
      "Amazon_Photo",
    }:
      Path(f"./synthetic_graphs/features").mkdir(parents=True, exist_ok=True)
      features = (
        torch.tensor(
          preprocess_features(
            np.load(
              (
                "./synthetic_graphs/features/{}/{}_{}.npy".format(
                  feature_base_name, feature_base_name, graph_idx
                )
              )
            )
          )
        ).clone().detach()
      )

    else:
      Path(f"./synthetic_graphs/features").mkdir(parents=True, exist_ok=True)
      features = (
        torch.tensor(
          preprocess_features(
            torch.load(
              (
                "./synthetic_graphs/features/{}/{}_{}.pt".format(
                  feature_base_name, feature_base_name, graph_idx
                )
              )
            ).detach().numpy()
          )
        ).clone().detach()
      )

    return adj, labels, degree, features


def preprocess_features(features):
  """
  Row-normalize feature matrix and convert to tuple representation
  """
  rowsum = np.array(features.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.0
  r_mat_inv = sp.diags(r_inv)
  features = r_mat_inv.dot(features)
  return features


def load_full_data(dataset_name):
    if dataset_name in {"cora_acm", "citeseer_acm", "pubmed_acm"}:
        adj, features, labels = load_data_cora(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()

    else:
        graph_adjacency_list_file_path = os.path.join(
            "../new_data", dataset_name, "out1_graph_edges.txt"
        )
        graph_node_features_and_labels_file_path = os.path.join(
            "../new_data", dataset_name, "out1_node_feature_label.txt"
        )

        G = nx.DiGraph().to_undirected()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == "film":
            with open(
                graph_node_features_and_labels_file_path
            ) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split("\t")
                    assert len(line) == 3
                    assert (
                        int(line[0]) not in graph_node_features_dict
                        and int(line[0]) not in graph_labels_dict
                    )
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(","), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(
                graph_node_features_and_labels_file_path
            ) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split("\t")
                    assert len(line) == 3
                    assert (
                        int(line[0]) not in graph_node_features_dict
                        and int(line[0]) not in graph_labels_dict
                    )
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(","), dtype=np.uint8
                    )
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split("\t")
                assert len(line) == 2
                if int(line[0]) not in G:
                    G.add_node(
                        int(line[0]),
                        features=graph_node_features_dict[int(line[0])],
                        label=graph_labels_dict[int(line[0])],
                    )
                if int(line[1]) not in G:
                    G.add_node(
                        int(line[1]),
                        features=graph_node_features_dict[int(line[1])],
                        label=graph_labels_dict[int(line[1])],
                    )
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))

        features = np.array(
            [
                features
                for _, features in sorted(G.nodes(data="features"), key=lambda x: x[0])
            ]
        )
        labels = np.array(
            [label for _, label in sorted(G.nodes(data="label"), key=lambda x: x[0])]
        )

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)  # .to(device)
    return adj, features, labels


def load_data_cora(dataset_str):
  """
  Loads input data from gcn/data directory
  ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
  ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
  ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
      (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
  ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
  ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
  ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
  ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
      object;
  ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
  All objects above must be saved using python pickle module.
  :param dataset_str: Dataset name
  :return: All data input files loaded (as well the training/test data).
  """
  names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
  objects = []
  if dataset_str == 'cora_acm':
    dataset_name = 'cora'
  if dataset_str == 'citeseer_acm':
    dataset_name = 'citeseer'
  if dataset_str == 'pubmed_acm':
    dataset_name = 'pubmed'
  for i in range(len(names)):
    # with open(f"../data/{dataset_str}/{dataset_str}/raw/ind.{}.{}".format(dataset_name, names[i]), "rb") as f:
    with open(f"../data/{dataset_str}/{dataset_str}/raw/ind.{dataset_name}.{names[i]}", "rb") as f:
      if sys.version_info > (3, 0):
        objects.append(pkl.load(f, encoding="latin1"))
      else:
        objects.append(pkl.load(f))

  x, y, tx, ty, allx, ally, graph = tuple(objects)
  # test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
  test_idx_reorder = parse_index_file(f"../data/{dataset_str}/{dataset_str}/raw/ind.{dataset_name}.test.index")
  test_idx_range = np.sort(test_idx_reorder)

  if dataset_name == "citeseer":
    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended

  features = sp.vstack((allx, tx)).tolil()
  features[test_idx_reorder, :] = features[test_idx_range, :]
  adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

  labels = np.vstack((ally, ty))
  labels[test_idx_reorder, :] = labels[test_idx_range, :]

  return adj, features, labels



def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
  """
  Convert a scipy sparse matrix to a torch sparse tensor.
  """
  sparse_mx = sparse_mx.tocoo().astype(np.float32)
  indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
  )
  values = torch.from_numpy(sparse_mx.data)
  shape = torch.Size(sparse_mx.shape)
  return torch.sparse.FloatTensor(indices, values, shape)

def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
      with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
        if sys.version_info > (3, 0):
          objects.append(pkl.load(f, encoding='latin1'))
        else:
          objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
      # Fix citeseer dataset (there are some isolated nodes in the graph)
      # Find isolated nodes, add them as zero-vecs into the right position
      test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
      tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
      tx_extended[test_idx_range - min(test_idx_range), :] = tx
      tx = tx_extended
      ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
      ty_extended[test_idx_range - min(test_idx_range), :] = ty
      ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
      features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test



def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)


def process(adj, features, normalize_adj, normalize_feats):
  if sp.isspmatrix(features):
    features = np.array(features.todense())
  if normalize_feats:
    features = normalize(features)
  features = torch.Tensor(features)
  if normalize_adj:
    adj = normalize(adj + sp.eye(adj.shape[0]))
  adj = sparse_mx_to_torch_sparse_tensor(adj)
  return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def augment(adj, features, normalize_feats=True):
  deg = np.squeeze(np.sum(adj, axis=0).astype(int))
  deg[deg > 5] = 5
  deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
  const_f = torch.ones(features.size(0), 1)
  features = torch.cat((features, deg_onehot, const_f), dim=1)
  return features