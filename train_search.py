import time
import torch
import random
import argparse
import numpy as np
import pickle as pkl
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data_prep import prepare_data
from utils import accuracy, EarlyStopping, consis_loss
from da_search import GCN_Framework, SAGE_Framework, GAT_Framework


def get_args():
    parser = argparse.ArgumentParser("gda-train-search")
    parser.add_argument("--dataset", type=str, default="cora", help="dataset name")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument("--drop_out", type=float, default=0.5, help="drop out rate")
    parser.add_argument("--hiddim", type=int, default=128, help="hidden dims")
    parser.add_argument(
        "--arch_learning_rate", type=float, default=0.08, help="arch learning rate"
    )
    parser.add_argument(
        "--gnn_learning_rate", type=float, default=0.01, help="gnn learning rate"
    )
    parser.add_argument(
        "--arch_weight_decay", type=float, default=5e-3, help="arch weight decay"
    )
    parser.add_argument(
        "--gnn_weight_decay", type=float, default=1e-3, help="gnn weight decay"
    )
    parser.add_argument(
        "--outer_epochs",
        type=int,
        default=100,
        help="num of training epochs for alphas",
    )
    parser.add_argument(
        "--aug_M", type=int, default=5, help="augmentation times per epoch"
    )
    parser.add_argument(
        "--basemodel", type=str, default="GCN", help="base network model"
    )
    parser.add_argument("--nlayers", type=int, default=2, help="layers")
    parser.add_argument(
        "--inner_epochs", type=int, default=300, help="inner update epochs"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="train gnn from scratch for epochs"
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="total training runs"
    )
    parser.add_argument(
        "--arch_learning_rate_min",
        type=float,
        default=0.01,
        help="arch learning rate min",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=0.8,
        help="max weight of regularization loss",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="confidence threshold for regularization loss",
    )
    parser.add_argument("--tem", type=float, default=0.1, help="sharpening temperature")
    parser.add_argument(
        "--reg_loss", type=str, default="l2", help="consistency loss function, l2 or kl"
    )
    parser.add_argument(
        "--use_gumbel_softmax", action="store_true", help="use gumbel_softmax"
    )
    ########################################################################
    parser.add_argument(
        "--label_num", type=int, default=20, help="number of labels per class"
    )
    parser.add_argument(
        "--edge_pt", type=float, default=1.0, help="remaining edge percent"
    )
    parser.add_argument(
        "--feature_pt", type=float, default=1.0, help="remaining feature percent"
    )
    args = parser.parse_args()
    return args


def init_pred_conf(num_nodes, labels, idx_train):
    pred = torch.zeros((num_nodes,)).type_as(labels)
    pred[idx_train] = labels[idx_train]
    conf = torch.zeros((num_nodes,))
    conf[idx_train] = 1.0

    return pred.cuda(), conf.cuda()


def main(
    run, args, graph, labels, index, num_nodes, nclass, nfeat, lg_s, node_s, aug_p=None
):
    idx_train, idx_val, idx_test = index
    idx_whole = set(torch.arange(num_nodes).tolist())
    unlabeled_idx = torch.LongTensor(
        list(idx_whole.difference(set(idx_train.tolist())))
    )

    if args.basemodel == "GCN":
        model = GCN_Framework(
            args, nfeat, args.hiddim, nclass, args.nlayers, F.relu, args.drop_out, aug_p
        )
    elif args.basemodel == "GraphSAGE":
        model = SAGE_Framework(
            args,
            nfeat,
            args.hiddim,
            nclass,
            args.nlayers,
            F.relu,
            args.drop_out,
            "gcn",
            aug_p,
        )
    else:
        args.gnn_learning_rate = 5e-3
        args.drop_out = 0.6
        args.hiddim = 16
        heads = ([8] * (args.nlayers - 1)) + [1]
        if args.dataset == "pubmed":
            heads[-1] = 8
        model = GAT_Framework(
            args,
            nfeat,
            args.hiddim,
            nclass,
            args.nlayers,
            F.elu,
            heads,
            args.drop_out,
            0.6,
            0.2,
            aug_p,
        )

    model = model.cuda()

    arch_optimizer = optim.Adam(
        model.arch_parameters(),
        lr=args.arch_learning_rate,
        weight_decay=args.arch_weight_decay,
    )

    scheduler_arch = optim.lr_scheduler.CosineAnnealingLR(
        arch_optimizer, args.outer_epochs, eta_min=args.arch_learning_rate_min
    )

    early_stopping_outer = EarlyStopping(
        patience=10, save_alphas=True, save_alphas_path=f"{args.dataset}_{args.basemodel}_eval_sp_best_alphas_{run}_ablation.pkl"
    )

    start_t = time.perf_counter()
    for epoch in range(args.outer_epochs):
        model._reset_parameters()
        total_loss = train_graph(
            args,
            graph,
            labels,
            idx_train,
            unlabeled_idx,
            num_nodes,
            lg_s,
            node_s,
            model,
            arch_optimizer,
        )
        scheduler_arch.step()

        early_stopping_outer(total_loss, model, epoch)

        if early_stopping_outer.early_stop:
            break

    end_t = time.perf_counter()
    total_t = end_t - start_t

    best_val_acc_list, final_test_acc_list = [], []

    best_alphas = pkl.load(open(early_stopping_outer.save_alphas_path, "rb"))
    for i, alpha in enumerate(best_alphas):
        model._arch_parameters[i] = Variable(alpha, requires_grad=False)

    for _ in range(10):  # 验证所选组合效果
        gnn_optimizer = optim.Adam(
            model.parameters(),
            lr=args.gnn_learning_rate,
            weight_decay=args.gnn_weight_decay,
        )

        model._reset_parameters()

        pred, conf = init_pred_conf(num_nodes, labels, idx_train)

        early_stopping_test = EarlyStopping(patience=30, use_loss=False)

        best_val_acc, final_test_acc = 0., 0.

        for epoch in range(args.epochs):
            pred, conf, val_acc, test_acc = retrain_graph(
                args,
                model,
                graph,
                labels,
                idx_train,
                idx_val,
                idx_test,
                unlabeled_idx,
                pred,
                conf,
                lg_s,
                node_s,
                epoch,
                gnn_optimizer,
                args.use_gumbel_softmax,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc

            early_stopping_test(val_acc, model, epoch)

            if early_stopping_test.early_stop:
                break

        best_val_acc_list.append(best_val_acc)
        final_test_acc_list.append(final_test_acc)

    mean_test_acc = np.mean(final_test_acc_list)

    return (
        mean_test_acc,
        np.std(final_test_acc_list),
        model._arch_parameters,
        total_t,
        np.mean(best_val_acc_list),
    )


def train_graph(
    args,
    graph,
    labels,
    idx_train,
    unlabeled_idx,
    num_nodes,
    lg_s,
    node_s,
    model,
    arch_optimizer,
):
    model_optimizer = optim.Adam(model.parameters(), lr=args.gnn_learning_rate)
    model.train()

    early_stopping_inner = EarlyStopping(
        patience=15, save_model=True, save_model_path=f"{args.dataset}_{args.basemodel}_eval_sp_inner_model_ablation.pt"
    )

    pred, conf = init_pred_conf(num_nodes, labels, idx_train)
    get_lambda = lambda t: args.max_lambda / args.inner_epochs * int((t + 1) / 5) * 5

    for ep in range(args.inner_epochs):
        model_optimizer.zero_grad()
        arch_optimizer.zero_grad()

        loss_M_sup = 0.0
        logits_unlabeled_M = []

        for _ in range(args.aug_M):
            logits, _ = model(
                graph, pred, conf, lg_s, node_s, ep, args.use_gumbel_softmax
            )
            loss_M_sup += F.nll_loss(logits[idx_train], labels[idx_train])
            logits_unlabeled_M.append(logits[unlabeled_idx])

        loss_M_sup /= args.aug_M
        loss_M_reg, average_logits = consis_loss(args, logits_unlabeled_M)

        pred[unlabeled_idx] = average_logits.max(1)[1].type_as(labels)
        pred[idx_train] = labels[idx_train]

        conf[unlabeled_idx] = average_logits.max(1)[0]
        conf[idx_train] = 1.0

        total_loss = loss_M_sup + get_lambda(ep) * loss_M_reg
        total_loss.backward()

        model_optimizer.step()

        early_stopping_inner(total_loss, model, ep)
        if early_stopping_inner.early_stop:
            break

    model.load_state_dict(torch.load(early_stopping_inner.save_model_path))

    model_optimizer.zero_grad()
    arch_optimizer.zero_grad()

    loss_M_sup = 0.
    logits_unlabeled_M = []
    best_epoch = early_stopping_inner.best_epoch
    for _ in range(args.aug_M):
        logits, _ = model(
            graph, pred, conf, lg_s, node_s, best_epoch, args.use_gumbel_softmax
        )
        loss_M_sup += F.nll_loss(logits[idx_train], labels[idx_train])
        logits_unlabeled_M.append(logits[unlabeled_idx])

    loss_M_sup /= args.aug_M
    loss_M_reg, average_logits = consis_loss(args, logits_unlabeled_M)

    total_loss = loss_M_sup + get_lambda(best_epoch) * loss_M_reg
    total_loss.backward()

    arch_optimizer.step()

    return total_loss


def retrain_graph(
    args,
    model,
    graph,
    labels,
    idx_train,
    idx_val,
    idx_test,
    unlabeled_idx,
    pred,
    conf,
    lg_s,
    node_s,
    epoch,
    gnn_optimizer,
    use_gumbel_softmax,
):
    model.train()
    gnn_optimizer.zero_grad()

    get_lambda = lambda t: args.max_lambda / args.inner_epochs * int((t + 1) / 5) * 5
    loss_M_sup = 0.0
    logits_unlabeled_M = []

    for _ in range(args.aug_M):
        logits, _ = model(
            graph,
            pred,
            conf,
            lg_s,
            node_s,
            epoch,
            args.use_gumbel_softmax,
            mode="evaluate_single_path",
        )
        loss_M_sup += F.nll_loss(logits[idx_train], labels[idx_train])
        logits_unlabeled_M.append(logits[unlabeled_idx])

    loss_M_sup /= args.aug_M
    loss_M_reg, average_logits = consis_loss(args, logits_unlabeled_M)

    pred[unlabeled_idx] = average_logits.max(1)[1].type_as(labels)
    pred[idx_train] = labels[idx_train]

    conf[unlabeled_idx] = average_logits.max(1)[0]
    conf[idx_train] = 1.0

    total_loss = loss_M_sup + get_lambda(epoch) * loss_M_reg
    total_loss.backward()

    gnn_optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(
            graph,
            pred,
            conf,
            lg_s,
            node_s,
            epoch,
            use_gumbel_softmax,
            mode="evaluate_single_path",
        )
    val_acc = accuracy(logits[idx_val], labels[idx_val])
    test_acc = accuracy(logits[idx_test], labels[idx_test])

    return pred, conf, val_acc, test_acc


if __name__ == "__main__":
    args = get_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(args.gpu)

    graph, labels, index, num_nodes, nclass, nfeat, lg_s, node_s = prepare_data(args)

    best_result = (0.0, 0.0)
    best_alphas = None
    total_t_list, best_ob_acc_list = [], []
    for run in range(args.runs):
        acc, std, alphas, total_t, best_ob_acc = main(
            run, args, graph, labels, index, num_nodes, nclass, nfeat, lg_s, node_s
        )
        total_t_list.append(total_t)
        best_ob_acc_list.append(best_ob_acc)
        print("Final best ten avg test acc: {:.4f}, std: {:.4f} ".format(acc, std))
        if acc > best_result[0]:
            best_result = (acc, std)
            best_alphas = alphas

    print(
        "\nBest result -- acc: {:.4f}, std: {:.4f}".format(
            best_result[0], best_result[1]
        )
    )
    print(best_alphas)
    print(
        f"Mean Total Time: {np.mean(total_t_list)}, Mean Best Observed Acc: {np.mean(best_ob_acc_list)}"
    )
