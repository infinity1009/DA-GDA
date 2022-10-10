import copy
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from utils import accuracy
from da_search import GCN_Framework, SAGE_Framework, GAT_Framework
from data_prep import prepare_data


def get_args():
    parser = argparse.ArgumentParser("gda-train-search")
    parser.add_argument('--dataset', type=str,
                        default='cora', help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')
    parser.add_argument('--drop_out', type=float,
                        default=0.5, help='drop out rate')
    parser.add_argument('--hiddim', type=int, default=128, help='hidden dims')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='learning rate')
    parser.add_argument('--arch_learning_rate', type=float,
                        default=0.08, help='arch learning rate')
    parser.add_argument('--gnn_learning_rate', type=float,
                        default=0.01, help='gnn learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--arch_weight_decay', type=float,
                        default=5e-3, help='arch weight decay')
    parser.add_argument('--gnn_weight_decay', type=float,
                        default=5e-4, help='gnn weight decay')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--basemodel', type=str, default='GCN',
                        help='base network model')
    parser.add_argument('--nlayers', type=int, default=2, help='layers')
    parser.add_argument('--w_update_epoch', type=int,
                        default=15, help='inner update epochs')
    parser.add_argument('--alpha_mode', type=str, default='train_loss',
                        help='choose loss type for alphas training')
    parser.add_argument('--learning_rate_min', type=float,
                        default=0.001, help='learning rate min')
    parser.add_argument('--arch_learning_rate_min', type=float,
                        default=0.005, help='arch learning rate min')
    parser.add_argument('--use_gumbel_softmax', action='store_true',
                        help='use gumbel_softmax')
########################################################################
    parser.add_argument('--label_num', type=int, default=20,
                        help='number of labels per class')
    parser.add_argument('--edge_pt', type=float, default=1.0,
                        help='remaining edge percent')
    parser.add_argument('--feature_pt', type=float, default=1.0,
                        help='remaining feature percent')
    args = parser.parse_args()
    return args


def main(args, graph, labels, index, nclass, nfeat, data, ebc, deg, evc, edge_x, aug_p=None):
    idx_train, idx_val, idx_test = index
    if args.basemodel == 'GCN':
        model = GCN_Framework(args, nfeat, args.hiddim, nclass,
                              args.nlayers, F.relu, args.drop_out, aug_p)
    elif args.basemodel == 'GraphSAGE':
        model = SAGE_Framework(args, nfeat, args.hiddim, nclass,
                               args.nlayers, F.relu, args.drop_out, 'gcn', aug_p)
    else:
        args.learning_rate = 5e-3
        args.drop_out = 0.6
        args.hiddim = 16
        heads = ([8]*(args.nlayers-1))+[1]
        if args.dataset == 'pubmed':
            heads[-1] = 8
        model = GAT_Framework(args, nfeat, args.hiddim, nclass,
                              args.nlayers, F.elu, heads, args.drop_out, 0.6, 0.2, aug_p)
    model = model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    arch_optimizer = optim.Adam(model.arch_parameters(),
                                lr=args.arch_learning_rate, weight_decay=args.arch_weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_arch = optim.lr_scheduler.CosineAnnealingLR(
        arch_optimizer, float(args.epochs), eta_min=args.arch_learning_rate_min)

    pred = torch.zeros_like(labels)  # initialization
    pred[idx_train] = labels[idx_train]

    conf = torch.zeros_like(labels, dtype=torch.float)
    conf[idx_train] = 1.

    for epoch in range(args.epochs):
        pred, conf = train_graph(
            args, graph, data, labels, idx_train, idx_val, pred, conf, ebc, deg, evc, edge_x, epoch, model, optimizer, arch_optimizer)
        scheduler.step()
        scheduler_arch.step()

    best_test_acc_list = []

    for _ in range(10):
        gnn_optimizer = optim.Adam(
            model.parameters(), lr=args.gnn_learning_rate, weight_decay=args.gnn_weight_decay)
        model._reset_parameters()

        pred = torch.zeros_like(labels)  # reinitialization
        pred[idx_train] = labels[idx_train]

        conf = torch.zeros_like(labels, dtype=torch.float)
        conf[idx_train] = 1.

        best_val_acc = 0.
        best_test_acc = 0.
        patience = 0
        for epoch in range(args.epochs):
            pred, conf, val_acc, test_acc = retrain_graph(
                model, graph, data, labels, idx_train, idx_val, idx_test, pred, conf, ebc, deg, evc, edge_x, epoch, gnn_optimizer, args.use_gumbel_softmax)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience = 0
            else:
                patience = patience + 1
                if patience > 30:
                    break

        best_test_acc_list.append(best_test_acc.item())

    return np.mean(best_test_acc_list)


def train_graph(args, graph, data, labels, idx_train, idx_val, pred, conf, ebc, deg, evc, edge_x, epoch, model, model_optimizer, arch_optimizer):
    model.train()

    for _ in range(args.w_update_epoch):
        model_optimizer.zero_grad()
        arch_optimizer.zero_grad()

        copy_graph = copy.deepcopy(graph)
        logits, _ = model(copy_graph, data, pred, conf, ebc,
                          deg, evc, edge_x, epoch, args.use_gumbel_softmax)
        loss = F.nll_loss(logits[idx_train], labels[idx_train])
        loss.backward()
        model_optimizer.step()

    model_optimizer.zero_grad()

    if args.alpha_mode == 'train_loss':
        arch_optimizer.step()
        copy_graph = copy.deepcopy(graph)
        logits, _ = model(copy_graph, data, pred, conf, ebc,
                          deg, evc, edge_x, epoch, args.use_gumbel_softmax)

    else:  # valid_loss
        arch_optimizer.zero_grad()
        copy_graph = copy.deepcopy(graph)
        logits, _ = model(copy_graph, data, pred, conf, ebc,
                          deg, evc, edge_x, epoch, args.use_gumbel_softmax)
        val_loss = F.nll_loss(logits[idx_val], labels[idx_val])
        val_loss.backward()
        arch_optimizer.step()

    pred = logits.max(1)[1].type_as(labels)
    pred[idx_train] = labels[idx_train]

    conf = logits.max(1)[0]
    conf = torch.exp(conf)
    conf[idx_train] = 1.

    return pred, conf


def retrain_graph(model, graph, data, labels, idx_train, idx_val, idx_test, pred, conf, ebc, deg, evc, edge_x, epoch, gnn_optimizer, use_gumbel_softmax):
    model.train()
    gnn_optimizer.zero_grad()
    copy_graph = copy.deepcopy(graph)

    logits, _ = model(copy_graph, data, pred, conf, ebc, deg, evc,
                      edge_x, epoch, use_gumbel_softmax, mode='evaluate_single_path')
    loss = F.nll_loss(logits[idx_train], labels[idx_train])
    loss.backward()
    gnn_optimizer.step()

    model.eval()
    copy_graph = copy.deepcopy(graph)
    with torch.no_grad():
        logits, _ = model(copy_graph, data, pred, conf, ebc, deg,
                          evc, edge_x, epoch, use_gumbel_softmax, mode='no aug')
    val_acc = accuracy(logits[idx_val], labels[idx_val])
    test_acc = accuracy(logits[idx_test], labels[idx_test])

    pred = logits.max(1)[1].type_as(labels)
    pred[idx_train] = labels[idx_train]

    conf = logits.max(1)[0]
    conf = torch.exp(conf)
    conf[idx_train] = 1.

    return pred, conf, val_acc, test_acc


if __name__ == "__main__":
    args = get_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(args.gpu)

    
    graph, labels, index, nclass, nfeat, data, ebc, deg, evc, edge_x = prepare_data(
            args, l=args.label_num, e=args.edge_pt, f=args.feature_pt)

    acc = main(args, graph, labels, index, nclass,
                   nfeat, data, ebc, deg, evc, edge_x)
    print("avg best test acc: {:.4f}".format(acc))
