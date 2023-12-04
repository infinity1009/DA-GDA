import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, add_self_loops



def sparse_convert(sparse_adj):
    row, col = sparse_adj._indices()
    edge_index = torch.stack((row, col), dim=0)
    return edge_index



def drop_node_weighted_mask(w, p: float, threshold: float = 0.5):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)

    drop_mask = torch.bernoulli(w).to(torch.bool)
    return drop_mask


def gen_node_mask(s, p: float = 0.1):
    q = torch.tensor([p], device=s.device)
    T = s.quantile(q)

    drop_mask = s < T
    return drop_mask


def drop_edge_weighted_mask(w, p: float, threshold: float = 0.5):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)

    drop_mask = torch.bernoulli(w).to(torch.bool)
    return drop_mask


def gen_edge_mask(s, p: float = 0.1):
    q = torch.tensor([p], device=s.device)
    T = s.quantile(q)
    
    drop_mask = s < T
    return drop_mask


def get_feature_weighted_mask(x_size, w, p: float, threshold: float = 0.5):
    w = w / w.mean() * p
    w = w.T.expand(x_size)
    w = w.where(w < threshold, torch.ones_like(w) * threshold)

    drop_mask = torch.bernoulli(w).to(torch.bool)
    return drop_mask


def feature_weights(x, node_c):
    w = torch.abs(x.t()) @ node_c

    s = torch.log(w)
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def sel_feature_weighted(x, w, p: float, threshold: float = 0.5):
    w = w / w.mean() * p
    w = w.T.expand(x.size())
    w = w.where(w < threshold, torch.ones_like(w) * threshold)

    sel_mask = torch.bernoulli(w).to(torch.bool)

    x = x.clone()
    x[sel_mask] = 1.

    return x

# node dropping--1


def node_dropping_1(N):
    drop_mask = torch.empty(
        (N,), dtype=torch.float32, device='cuda').uniform_(0, 1) < 0.05
    return drop_mask

# node dropping--2


def node_dropping_2(N):
    drop_mask = torch.empty(
        (N,), dtype=torch.float32, device='cuda').uniform_(0, 1) < 0.1
    return drop_mask

# node dropping--3


def node_dropping_3(N):
    drop_mask = torch.empty(
        (N,), dtype=torch.float32, device='cuda').uniform_(0, 1) < 0.15
    return drop_mask


# node dropping--4


def node_dropping_4(deg):
    # degree
    s = torch.log(deg)
    weights = (s.max() - s) / (s.max() - s.mean())
    drop_mask = drop_node_weighted_mask(weights, 0.05, 0.1)
    return drop_mask

# node dropping--5


def node_dropping_5(deg):
   # degree
    s = torch.log(deg)
    weights = (s.max() - s) / (s.max() - s.mean())
    drop_mask = drop_node_weighted_mask(weights, 0.1, 0.15)
    return drop_mask

# node dropping--6


def node_dropping_6(node_s):
    # node feature mlp
    drop_mask = gen_node_mask(node_s, 0.1)
    return drop_mask

# node dropping--7


def node_dropping_7(node_s):
    # node feature mlp
    drop_mask = gen_node_mask(node_s, 0.15)
    return drop_mask


# edge dropping--1

def edge_dropping_1(edge_index):
    drop_mask = torch.rand(edge_index.size(1), device=edge_index.device) < 0.1
    return drop_mask

# edge dropping--2


def edge_dropping_2(edge_index):
    drop_mask = torch.rand(edge_index.size(1), device=edge_index.device) < 0.15
    return drop_mask

# edge dropping--3


def edge_dropping_3(edge_index):
    drop_mask = torch.rand(edge_index.size(1), device=edge_index.device) < 0.2
    return drop_mask

# edge dropping--4


def edge_dropping_4(edge_index):
    drop_mask = torch.rand(edge_index.size(1), device=edge_index.device) < 0.25
    return drop_mask

# edge dropping--5


def edge_dropping_5(edge_index):
    drop_mask = torch.rand(edge_index.size(1), device=edge_index.device) < 0.3
    return drop_mask


# # edge dropping--6

# def edge_dropping_6(ebc):
#     # edge betweenness centrality
#     weights = (ebc.max() - ebc) / (ebc.max() - ebc.mean())
#     drop_mask = drop_edge_weighted_mask(weights, 0.1, 0.15)

#     return drop_mask


# # edge dropping--7


# def edge_dropping_7(ebc):
#     # edge betweenness centrality
#     weights = (ebc.max() - ebc) / (ebc.max() - ebc.mean())
#     drop_mask = drop_edge_weighted_mask(weights, 0.2, 0.3)

#     return drop_mask

# edge dropping--6

def edge_dropping_6(deg, edge_index):
    # mean degree centrality
    row, col = edge_index
    s = (deg[row] + deg[col]) / 2
    s = torch.log(s)
    weights = (s.max() - s) / (s.max() - s.mean())
    drop_mask = drop_edge_weighted_mask(weights, 0.1, 0.15)

    return drop_mask


# edge dropping--7


def edge_dropping_7(deg, edge_index):
    # mean degree centrality
    row, col = edge_index
    s = (deg[row] + deg[col]) / 2
    s = torch.log(s)
    weights = (s.max() - s) / (s.max() - s.mean())
    drop_mask = drop_edge_weighted_mask(weights, 0.2, 0.3)

    return drop_mask

# edge dropping--8


def edge_dropping_8(lg_s):
    drop_mask = gen_edge_mask(lg_s, 0.15)

    return drop_mask

# edge dropping--9


def edge_dropping_9(lg_s):
    drop_mask = gen_edge_mask(lg_s, 0.25)

    return drop_mask

# edge dropping--10


def edge_dropping_10(edge_index, num_nodes, pred, conf, T, epoch, t):
    # update conf_min every T epochs (total t times), conf_min starts with conf_st, ends with cond_ed
    conf_st, conf_ed = 0.8, 0.95
    conf_min = conf_st + int(epoch/T)*(conf_ed-conf_st)/t

    pred_0, pred_1 = pred[edge_index[0]], pred[edge_index[1]]
    conf_0, conf_1 = (conf[edge_index[0]] >= conf_min).to(
        torch.bool), (conf[edge_index[1]] >= conf_min).to(torch.bool)
    candidate = (pred_0 != pred_1).nonzero(
        as_tuple=False).view(-1)  # candidate inter-class edges' ids
    confident = (conf_0 & conf_1).nonzero(
        as_tuple=False).view(-1)  # confident edges

    drop_mask_1 = torch.zeros(
        (edge_index.shape[1],), device='cuda').to(torch.bool)
    drop_mask_1[candidate] = True
    drop_mask_2 = torch.zeros(
        (edge_index.shape[1],), device='cuda').to(torch.bool)
    drop_mask_2[confident] = True
    drop_mask = drop_mask_1 & drop_mask_2

    idx = drop_mask.nonzero(as_tuple=False).view(-1)
    idx = idx[torch.randperm(idx.size(0))]

    # control the magnitude of dropping
    p = 0.05
    if num_nodes == 1190:
        p = 0.005

    upper_bound = min(int(idx.size(0)*0.5), int(edge_index.shape[1]*p))
    idx = idx[:upper_bound]
    drop_mask[:] = False
    drop_mask[idx] = True

    return drop_mask

# edge dropping--11


def edge_dropping_11(edge_index, num_nodes, pred, conf, T, epoch, t):
    # update conf_min every T epochs (total t times), conf_min starts with conf_st, ends with cond_ed
    conf_st, conf_ed = 0.75, 0.95
    conf_min = conf_st + int(epoch/T)*(conf_ed-conf_st)/t

    pred_0, pred_1 = pred[edge_index[0]], pred[edge_index[1]]
    conf_0, conf_1 = (conf[edge_index[0]] >= conf_min).to(
        torch.bool), (conf[edge_index[1]] >= conf_min).to(torch.bool)
    candidate = (pred_0 != pred_1).nonzero(
        as_tuple=False).view(-1)  # candidate inter-class edges' ids
    confident = (conf_0 & conf_1).nonzero(
        as_tuple=False).view(-1)  # confident edges

    drop_mask_1 = torch.zeros(
        (edge_index.shape[1],), device='cuda').to(torch.bool)
    drop_mask_1[candidate] = True
    drop_mask_2 = torch.zeros(
        (edge_index.shape[1],), device='cuda').to(torch.bool)
    drop_mask_2[confident] = True
    drop_mask = drop_mask_1 & drop_mask_2

    idx = drop_mask.nonzero(as_tuple=False).view(-1)
    idx = idx[torch.randperm(idx.size(0))]

    # control the magnitude of dropping
    p = 0.05
    if num_nodes == 1190:
        p = 0.005

    upper_bound = min(int(idx.size(0)*0.5), int(edge_index.shape[1]*p))
    idx = idx[:upper_bound]
    drop_mask[:] = False
    drop_mask[idx] = True

    return drop_mask


# attribute perturbing--1


def attr_perturbing_1(x):
    sign_x = torch.sign(x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()

    drop_mask = torch.empty(
        (x.shape[1],), dtype=torch.float32, device='cuda').uniform_(0, 1) < 0.05 * attr_density
    drop_mask = drop_mask.unsqueeze(1)
    drop_mask = drop_mask.expand(x.shape[1], x.shape[0]).T
    return drop_mask

# attribute perturbing--2


def attr_perturbing_2(x):
    sign_x = torch.sign(x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()

    drop_mask = torch.empty(
        (x.shape[1],), dtype=torch.float32, device='cuda').uniform_(0, 1) < 0.1 * attr_density
    drop_mask = drop_mask.unsqueeze(1)
    drop_mask = drop_mask.expand(x.shape[1], x.shape[0]).T
    return drop_mask

# attribute perturbing--3


def attr_perturbing_3(x):
    sign_x = torch.sign(x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()

    drop_mask = torch.empty(
        (x.shape[1],), dtype=torch.float32, device='cuda').uniform_(0, 1) < 0.15 * attr_density
    drop_mask = drop_mask.unsqueeze(1)
    drop_mask = drop_mask.expand(x.shape[1], x.shape[0]).T
    return drop_mask


# attribute perturbing--4


def attr_perturbing_4(x, node_deg):
    sign_x = torch.sign(x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()   
    drop_feature_weights = feature_weights(x, node_c=node_deg)

    return get_feature_weighted_mask(x.size(), drop_feature_weights, 0.1 * attr_density, 0.15 * attr_density)

# attribute perturbing--5


def attr_perturbing_5(x, node_deg):
    sign_x = torch.sign(x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()
    drop_feature_weights = feature_weights(x, node_c=node_deg)
    
    return get_feature_weighted_mask(x.size(), drop_feature_weights, 0.15 * attr_density, 0.2 * attr_density)

# attribute perturbing--6


def attr_perturbing_6(x, node_s):
    sign_x = torch.sign(x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()
    drop_feature_weights = feature_weights(x, node_c=node_s)

    return get_feature_weighted_mask(x.size(), drop_feature_weights, 0.1 * attr_density, 0.15 * attr_density)

# attribute perturbing--7


def attr_perturbing_7(x, node_s):
    sign_x = torch.sign(x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()
    drop_feature_weights = feature_weights(x, node_c=node_s)

    return get_feature_weighted_mask(x.size(), drop_feature_weights, 0.15 * attr_density, 0.2 * attr_density)

# attribute adding--1


def attr_adding_1(x, edge_index, edge_weight):
    N, f = x.shape[0], x.shape[1]
    duo_index, duo_weight = to_undirected(edge_index, edge_weight)
    duo_index, duo_weight = add_self_loops(
        duo_index, duo_weight, num_nodes=N)
    deg = torch.zeros((N, ), dtype=torch.float,
                      device=duo_index.device)
    deg = deg.scatter_add_(0, duo_index[0], duo_weight).unsqueeze(1)
    deg = deg.expand(N, f)
    value = torch.ones_like(duo_index[0], device='cuda', dtype=torch.float)

    duo_index = SparseTensor(
        row=duo_index[0], col=duo_index[1], value=value, sparse_sizes=(N, N))
    out = duo_index.matmul(x)
    out = out / deg

    x = 0.8*x + 0.2*out

    return x

# attribute adding--2


def attr_adding_2(x, edge_index, edge_weight):
    abs_x = torch.abs(x)
    sign_x = torch.sign(abs_x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()
    per_density = sign_x.sum(dim=1) / sign_x.shape[1]

    ones = torch.ones((x.shape[0],), device='cuda').to(torch.bool)
    zeros = torch.zeros((x.shape[0],), device='cuda').to(torch.bool)

    # augment nodes with extremly sparse features (maybe missing attribute)
    sel_sparse = torch.where(per_density < 0.2*attr_density, ones, zeros)
    sparse_node = torch.arange(x.shape[0])[sel_sparse]

    N = x.shape[0]
    duo_index, duo_weight = to_undirected(edge_index, edge_weight)
    duo_index, duo_weight = add_self_loops(
        duo_index, duo_weight, num_nodes=N)
    deg = torch.zeros((N, ), dtype=torch.float,
                      device=duo_index.device)
    deg = deg.scatter_add_(0, duo_index[0], duo_weight).unsqueeze(1)
    deg = deg.expand(x.size())
    value = torch.ones_like(duo_index[0], device='cuda', dtype=torch.float)

    duo_index = SparseTensor(
        row=duo_index[0], col=duo_index[1], value=value, sparse_sizes=(N, N))
    out = duo_index.matmul(x)
    out = out / deg

    x = x.clone()
    x[sparse_node, :] = out[sparse_node]

    return x

# attribute adding--3


def attr_adding_3(x):
    # add Gaussian-Noise N(0, 1)
    x = x + torch.randn(x.size(), device='cuda')

    return x

# attribute adding--4


def attr_adding_4(x):
    abs_x = torch.abs(x)
    sign_x = torch.sign(abs_x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()
    sel_mask = torch.empty(
        x.shape, dtype=torch.float32, device='cuda').uniform_(0, 1) < 0.1 * attr_density
    x = x.clone()
    x[sel_mask] = 1

    return x

# attribute adding--5


def attr_adding_5(x):
    abs_x = torch.abs(x)
    sign_x = torch.sign(abs_x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()
    sel_mask = torch.empty(
        x.shape, dtype=torch.float32, device='cuda').uniform_(0, 1) < 0.15 * attr_density
    x = x.clone()
    x[sel_mask] = 1

    return x

# attribute adding--6


def attr_adding_6(x):
    abs_x = torch.abs(x)
    sign_x = torch.sign(abs_x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()
    sel_mask = torch.empty(
        x.shape, dtype=torch.float32, device='cuda').uniform_(0, 1) < 0.2 * attr_density
    x = x.clone()
    x[sel_mask] = 1

    return x

# attribute adding--7


def attr_adding_7(x, edge_index, edge_weight):
    abs_x = torch.abs(x)
    sign_x = torch.sign(abs_x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()

    N = x.shape[0]
    duo_index, duo_weight = to_undirected(edge_index, edge_weight)
    duo_index, duo_weight = add_self_loops(
        duo_index, duo_weight, num_nodes=N)
    node_deg = torch.zeros((N, ), dtype=torch.float,
                           device=duo_index.device)
    node_deg = node_deg.scatter_add_(0, duo_index[0], duo_weight).unsqueeze(1)
    # dropped nodes don't affect feature weights as their features are set to zero
    sel_feature_weights = feature_weights(x, node_c=node_deg)
    return sel_feature_weighted(x, sel_feature_weights, 0.1 * attr_density, 0.15 * attr_density)

# attribute adding--8


def attr_adding_8(x, edge_index, edge_weight):
    abs_x = torch.abs(x)
    sign_x = torch.sign(abs_x).float()
    attr_density = (sign_x.sum().sum() /
                    (sign_x.shape[0] * sign_x.shape[1])).item()

    N = x.shape[0]
    duo_index, duo_weight = to_undirected(edge_index, edge_weight)
    duo_index, duo_weight = add_self_loops(
        duo_index, duo_weight, num_nodes=N)
    node_deg = torch.zeros((N, ), dtype=torch.float,
                           device=duo_index.device)
    node_deg = node_deg.scatter_add_(0, duo_index[0], duo_weight).unsqueeze(1)
    # dropped nodes don't affect feature weights as their features are set to zero
    sel_feature_weights = feature_weights(x, node_c=node_deg)
    return sel_feature_weighted(x, sel_feature_weights, 0.15 * attr_density, 0.2 * attr_density)


node_minus = [None, node_dropping_1, node_dropping_2,
              node_dropping_3, node_dropping_4, node_dropping_5, node_dropping_6, node_dropping_7]
attr_plus = [None, attr_adding_1, attr_adding_2, attr_adding_3,
             attr_adding_4, attr_adding_5, attr_adding_6, attr_adding_7, attr_adding_8]
attr_perturb = [None, attr_perturbing_1, attr_perturbing_2,
                attr_perturbing_3, attr_perturbing_4, attr_perturbing_5, attr_perturbing_6, attr_perturbing_7]
edge_minus = [None, edge_dropping_1, edge_dropping_2,
              edge_dropping_3, edge_dropping_4, edge_dropping_5, edge_dropping_6, edge_dropping_7, edge_dropping_10, edge_dropping_11, edge_dropping_8, edge_dropping_9]
aug_pipeline = [edge_minus, attr_perturb, attr_plus, node_minus]
aug_p1 = [edge_minus]
aug_p2 = [attr_perturb, attr_plus]
aug_p3 = [node_minus]
aug_p4 = [edge_minus, attr_perturb, attr_plus]
aug_p5 = [attr_perturb, attr_plus, node_minus]
aug_p6 = [edge_minus, node_minus]
pipelines = [aug_p1, aug_p2, aug_p3, aug_p4, aug_p5, aug_p6]
