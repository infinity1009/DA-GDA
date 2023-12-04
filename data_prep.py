import sys
import torch
import numpy as np
import pickle as pkl
import networkx as nx
import os.path as Path
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.datasets import Twitch
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected


def load_citation_data(path="../data/raw_data", dataset="citeseer"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    nclass = ally.shape[1]

    test_idx_reorder = parse_index_file(
        "{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # type(features): scipy.sparse.lil.lil_matrix
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    nfeat = features.shape[1]

    # type(adj): scipy.sparse.csr.csr_matrix
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # type(features): torch.Tensor
    features = torch.FloatTensor(np.array(features.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    idx_train, idx_val, idx_test = list(
        map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test, nclass, nfeat


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def convert_adj(adj):
    row, col = adj
    mask = torch.ones((adj.shape[1],)).to(torch.bool)
    mask[row >= col] = False
    return adj[:, mask]


def load_binary_data(ds):
    tvt_nids = pkl.load(open(f'../data/graphs/{ds}_tvt_nids.pkl', 'rb'))
    adj = pkl.load(open(f'../data/graphs/{ds}_adj.pkl', 'rb'))
    features = pkl.load(open(f'../data/graphs/{ds}_features.pkl', 'rb'))
    labels = pkl.load(open(f'../data/graphs/{ds}_labels.pkl', 'rb'))

    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    nfeat = features.shape[1]

    labels = torch.LongTensor(labels)
    nclass = len(torch.unique(labels))

    train_nid = tvt_nids[0]
    val_nid = tvt_nids[1]
    test_nid = tvt_nids[2]

    adj = from_scipy_sparse_matrix(adj)[0]
    return adj, features, labels, train_nid, val_nid, test_nid, nclass, nfeat


def prepare_data(args):
    l = args.label_num
    e = args.edge_pt
    f = args.feature_pt

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test, nclass, nfeat = load_citation_data(
            path='../data/raw_data', dataset=args.dataset)

        if l < 20:  # label sparsity
            file_name = f'sparse_graph/{args.dataset}_tvt_nids_{l}.pkl'
            idx_train = pkl.load(open(file_name, 'rb'))[0]

        if e < 1.0:
            file_name = f'sparse_graph/{args.dataset}_adj_{e}.pkl'
            adj = pkl.load(open(file_name, 'rb'))

        if f < 1.0:
            file_name = f'sparse_graph/{args.dataset}_features_{f}.pkl'
            features = pkl.load(open(file_name, 'rb'))

        adj = sp.triu(adj, k=1)  # upper triangular portion of adj
        row, col = adj.nonzero()
        row, col = torch.LongTensor(row), torch.LongTensor(col)
        adj = torch.stack((row, col), dim=0)  # edge_index format    

    elif args.dataset.startswith("Twitch"):
        name = args.dataset.split("_")[1]
        dataset = Twitch(root="../data", name=name)       
        data = dataset[0]
        adj, features, labels, nclass, nfeat = data.edge_index, data.x, data.y, dataset.num_classes, data.num_features
        idx_train = pkl.load(open(f'../data/{name}/idx_train.pkl', 'rb'))
        idx_val = pkl.load(open(f'../data/{name}/idx_val.pkl', 'rb'))
        idx_test = pkl.load(open(f'../data/{name}/idx_test.pkl', 'rb'))
        
        features = F.normalize(features)
        adj = convert_adj(adj)

    else:
        adj, features, labels, train_mask, val_mask, test_mask, nclass, nfeat = load_binary_data(
            args.dataset)
        index = torch.arange(features.shape[0])
        idx_train, idx_val, idx_test = index[train_mask], index[val_mask], index[test_mask]

        adj = to_undirected(adj, num_nodes=features.shape[0])
        adj = convert_adj(adj)

    graph = adj.cuda(), features.cuda()
    labels = labels.cuda()
    index = idx_train.cuda(), idx_val.cuda(), idx_test.cuda()

    lg_s = pkl.load(open(f'./metric/{args.dataset}_lg_s.pkl', 'rb')).cuda()
    
    node_s = pkl.load(open(f'./metric/{args.dataset}_node_s.pkl', 'rb')).squeeze(1).cuda()

    return graph, labels, index, features.size(0), nclass, nfeat, lg_s, node_s