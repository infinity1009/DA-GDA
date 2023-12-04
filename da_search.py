import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.utils import to_undirected, add_self_loops
from dgl.nn.pytorch.conv import GraphConv, SAGEConv, GATConv

from search_space import aug_pipeline
from utils import take_augmentation, add_self_loops_mean

class GCN_Framework(nn.Module):  # include a supernet of data augmentation and GNN

    def __init__(self, args, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aug_p):
        super(GCN_Framework, self).__init__()

        self.args = args
        self.nclass = n_classes
        self.dropout = nn.Dropout(p=dropout)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(
            GraphConv(in_feats, n_hidden, activation=activation))
        for _ in range(n_layers - 2):
            self.gnn_layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation))
        self.gnn_layers.append(GraphConv(n_hidden, n_classes))
        if aug_p is None:
            aug_p = aug_pipeline
        self.aug_p = aug_p

        self._initialize_alphas()

    def _initialize_alphas(self):
        self._arch_parameters = []

        for ops in self.aug_p:
            num_ops = len(ops)
            log_alphas = Variable(
                1e-2*torch.randn(num_ops).cuda(), requires_grad=True)
            self._arch_parameters.append(log_alphas)

    def _reset_parameters(self):
        for gnn_layer in self.gnn_layers:
            gnn_layer.reset_parameters()

    def arch_parameters(self):
        return self._arch_parameters

    def get_one_hot_alpha(self, alpha):
        one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
        idx = torch.argmax(alpha, dim=-1)

        value = torch.ones_like(idx, device=alpha.device).to(torch.float)
        one_hot_alpha = one_hot_alpha.index_put((idx, ), value)

        return one_hot_alpha
    
    def show_policy(self):
        ops = []
        
        for op_list, alpha in zip(self.aug_p, self._arch_parameters):
            one_hot_alpha = self.get_one_hot_alpha(alpha)
            idx = torch.argmax(one_hot_alpha)
            ops.append(op_list[idx])
        
        return ops

    def forward(self, graph, pred=None, conf=None, lg_s=None, node_s=None, epoch=50, use_gumbel_softmax=False, tau=0.5, mode='none'):
        edge_index, x = graph

        alphas_list = []
        for i, log_alphas in enumerate(self._arch_parameters):
            if not use_gumbel_softmax:
                alphas = F.softmax(log_alphas, dim=-1)
            else:
                alphas = F.gumbel_softmax(log_alphas, tau=tau)
            if mode == 'evaluate_single_path':
                alphas = self.get_one_hot_alpha(alphas)
            alphas_list.append(alphas)

        edge_weights = torch.ones(
            edge_index.size(1), device=edge_index.device).float()

        if mode != 'no aug':
            for i in range(len(self.aug_p)):
                aug_ops = self.aug_p[i]
                alphas = alphas_list[i]
                edge_weights, x = take_augmentation(
                    aug_ops, x, edge_weights, edge_index, alphas, lg_s, node_s, pred, conf, 10, epoch, 20, mode)

        if mode == 'evaluate_single_path':
            mask = edge_weights.nonzero(as_tuple=False).view(-1)
            edge_index = edge_index[:, mask]
            edge_weights = edge_weights[mask]

        edge_index, edge_weights = to_undirected(edge_index, edge_weights)
        edge_index, edge_weights = add_self_loops_mean(
            edge_index, edge_weights, x.shape[0])
        mask = (edge_weights == 0.).nonzero(as_tuple=False).view(-1)
        edge_weights[mask] = 1.0  # for isolated nodes
        g = dgl.graph((edge_index[0], edge_index[1]))

        if x.shape[1] not in [8189, 128]:
            x = x / x.sum(1, keepdim=True).clamp(min=1)
        h = x
        for i, layer in enumerate(self.gnn_layers):
            if i > 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weight=edge_weights)

        return F.log_softmax(h, dim=1), alphas_list


class SAGE_Framework(nn.Module):  # include a supernet of data augmentation and GNN

    def __init__(self, args, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type, aug_p):
        super(SAGE_Framework, self).__init__()

        self.args = args
        self.nclass = n_classes
        self.dropout = nn.Dropout(p=dropout)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(SAGEConv(
            in_feats, n_hidden, aggregator_type, feat_drop=0., activation=activation))
        for _ in range(n_layers - 2):
            self.gnn_layers.append(SAGEConv(
                n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        self.gnn_layers.append(SAGEConv(
            n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))
        if aug_p is None:
            aug_p = aug_pipeline
        self.aug_p = aug_p

        self._initialize_alphas()

    def _initialize_alphas(self):
        self._arch_parameters = []

        for ops in self.aug_p:
            num_ops = len(ops)
            log_alphas = Variable(
                1e-2*torch.randn(num_ops).cuda(), requires_grad=True)
            self._arch_parameters.append(log_alphas)

    def _reset_parameters(self):
        for gnn_layer in self.gnn_layers:
            gnn_layer.reset_parameters()

    def arch_parameters(self):
        return self._arch_parameters

    def get_one_hot_alpha(self, alpha):
        one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
        idx = torch.argmax(alpha, dim=-1)

        value = torch.ones_like(idx, device=alpha.device).to(torch.float)
        one_hot_alpha = one_hot_alpha.index_put((idx, ), value)

        return one_hot_alpha

    def show_policy(self):
        ops = []
        
        for op_list, alpha in zip(self.aug_p, self._arch_parameters):
            one_hot_alpha = self.get_one_hot_alpha(alpha)
            idx = torch.argmax(one_hot_alpha)
            ops.append(op_list[idx])
        
        return ops
    
    def forward(self, graph, pred, conf, lg_s, node_s, epoch, use_gumbel_softmax=False, tau=0.5, mode='none'):
        edge_index, x = graph

        alphas_list = []
        for i, log_alphas in enumerate(self._arch_parameters):
            if not use_gumbel_softmax:
                alphas = F.softmax(log_alphas, dim=-1)
            else:
                alphas = F.gumbel_softmax(log_alphas, tau=tau)
            if mode == 'evaluate_single_path':
                alphas = self.get_one_hot_alpha(alphas)
            alphas_list.append(alphas)

        edge_weights = torch.ones(
            edge_index.size()[1], device=edge_index.device).float()

        if mode != 'no aug':
            for i in range(len(self.aug_p)):
                aug_ops = self.aug_p[i]
                alphas = alphas_list[i]
                edge_weights, x = take_augmentation(
                    aug_ops, x, edge_weights, edge_index, alphas, lg_s, node_s, pred, conf, 10, epoch, 20, mode)

        if mode == 'evaluate_single_path':
            mask = edge_weights.nonzero(as_tuple=False).view(-1)
            edge_index = edge_index[:, mask]
            edge_weights = edge_weights[mask]

        edge_index, edge_weights = to_undirected(edge_index, edge_weights)
        edge_index, edge_weights = add_self_loops_mean(
            edge_index, edge_weights, x.shape[0])
        mask = (edge_weights == 0.).nonzero(as_tuple=False).view(-1)
        edge_weights[mask] = 1.0  # for isolated nodes
        g = dgl.graph((edge_index[0], edge_index[1]))

        if x.shape[1] != 8189:
            x = x / x.sum(1, keepdim=True).clamp(min=1)
        h = x
        for layer in self.gnn_layers:
            h = layer(g, h, edge_weight=edge_weights)

        return F.log_softmax(h, dim=1), alphas_list


class GAT_Framework(nn.Module):  # include a supernet of data augmentation and GNN

    def __init__(self, args, in_feats, n_hidden, n_classes, n_layers, activation, heads,
                 dropout,
                 attn_drop,
                 negative_slope, aug_p):
        super(GAT_Framework, self).__init__()

        self.args = args
        self.nclass = n_classes
        self.dropout = nn.Dropout(p=dropout)

        self.n_layers = n_layers
        self.gnn_layers = nn.ModuleList()
        # input layer
        self.gnn_layers.append(GATConv(
            in_feats, n_hidden, heads[0], dropout, attn_drop, negative_slope, False, activation=activation))
        # hidden layers
        for i in range(n_layers - 2):
            self.gnn_layers.append(GATConv(
                n_hidden * heads[i], n_hidden, heads[i+1], dropout, attn_drop, negative_slope, False, activation=activation))
        # output layer
        self.gnn_layers.append(GATConv(
            n_hidden * heads[-2], n_classes, heads[-1], dropout, attn_drop, negative_slope, False, activation=None))
        if aug_p is None:
            aug_p = aug_pipeline
        self.aug_p = aug_p

        self._initialize_alphas()

    def _initialize_alphas(self):
        self._arch_parameters = []
        
        for ops in self.aug_p:
            num_ops = len(ops)
            log_alphas = Variable(
                1e-2*torch.randn(num_ops).cuda(), requires_grad=True)
            self._arch_parameters.append(log_alphas)

    def _reset_parameters(self):
        for gnn_layer in self.gnn_layers:
            gnn_layer.reset_parameters()

    def arch_parameters(self):
        return self._arch_parameters

    def get_one_hot_alpha(self, alpha):
        one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
        idx = torch.argmax(alpha, dim=-1)

        value = torch.ones_like(idx, device=alpha.device).to(torch.float)
        one_hot_alpha = one_hot_alpha.index_put((idx, ), value)

        return one_hot_alpha

    def show_policy(self):
        ops = []
        
        for op_list, alpha in zip(self.aug_p, self._arch_parameters):
            one_hot_alpha = self.get_one_hot_alpha(alpha)
            idx = torch.argmax(one_hot_alpha)
            ops.append(op_list[idx])
        
        return ops
    
    def forward(self, graph, pred, conf, lg_s, node_s, epoch, use_gumbel_softmax=False, tau=0.5, mode='none'):
        edge_index, x = graph

        alphas_list = []
        for log_alphas in self._arch_parameters:
            if not use_gumbel_softmax:
                alphas = F.softmax(log_alphas, dim=-1)
            else:
                alphas = F.gumbel_softmax(log_alphas, tau=tau)
            if mode == 'evaluate_single_path':
                alphas = self.get_one_hot_alpha(alphas)
            alphas_list.append(alphas)

        edge_weights = torch.ones(
            edge_index.size()[1], device=edge_index.device).float()

        if mode != 'no aug':
            for i in range(len(self.aug_p)):
                aug_ops = self.aug_p[i]
                alphas = alphas_list[i]
                edge_weights, x = take_augmentation(
                    aug_ops, x, edge_weights, edge_index, alphas, lg_s, node_s, pred, conf, 10, epoch, 20, mode)

        if mode == 'evaluate_single_path':
            mask = edge_weights.nonzero(as_tuple=False).view(-1)
            edge_index = edge_index[:, mask]
            edge_weights = edge_weights[mask]

        edge_index = to_undirected(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.shape[0])[0]
        g = dgl.graph((edge_index[0], edge_index[1]))

        if x.shape[1] != 8189:
            x = x / x.sum(1, keepdim=True).clamp(min=1)
        h = x
        for l in range(self.n_layers-1):
            h = self.gnn_layers[l](g, h).flatten(1)
        logits = self.gnn_layers[-1](g, h).mean(1)

        return F.log_softmax(logits, dim=1), alphas_list
