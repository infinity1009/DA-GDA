import numpy as np
import pickle as pkl
from sklearn.metrics import f1_score
from search_space import *
from torch_scatter import scatter


def mask4edge(edge_index, drop_mask):
    sel_mask_1, sel_mask_2 = edge_index[0], edge_index[1]
    zero_tensor = torch.zeros((edge_index.shape[1],), device="cuda")
    one_tensor = torch.ones((edge_index.shape[1],), device="cuda")
    sel_mask_1 = torch.where(drop_mask[sel_mask_1], zero_tensor, one_tensor).to(
        torch.bool
    )
    sel_mask_2 = torch.where(drop_mask[sel_mask_2], zero_tensor, one_tensor).to(
        torch.bool
    )
    sel_mask = sel_mask_1 & sel_mask_2
    return sel_mask


def add_self_loops_mean(edge_index, edge_attr, N):
    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N, reduce="mean")
    edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_attr


def take_augmentation(
    aug_ops,
    x,
    edge_weights,
    edge_index,
    alphas,
    lg_s,
    node_s,
    pred,
    conf,
    T,
    epoch,
    t,
    mode,
):
    N = x.shape[0]
    if aug_ops is node_minus:
        final_mask = torch.zeros((N,), device=x.device)
        final_edge_mask = torch.zeros((edge_index.shape[1],), device=x.device)
        for i in range(len(aug_ops)):
            aug_op = aug_ops[i]
            alpha = alphas[i]
            if aug_op is None:
                drop_mask = torch.zeros((N,), dtype=torch.bool, device=x.device)
            elif aug_op in [node_dropping_1, node_dropping_2, node_dropping_3]:
                drop_mask = aug_op(N)
            elif aug_op in [node_dropping_4, node_dropping_5]:
                bi_index, bi_weights = to_undirected(
                    edge_index, edge_weights, num_nodes=N
                )
                bi_index, bi_weights = add_self_loops(bi_index, bi_weights, num_nodes=N)
                deg = scatter(bi_weights, bi_index[0])
                drop_mask = aug_op(deg)
            else:
                drop_mask = aug_op(node_s)
            edge_sel_mask = mask4edge(edge_index, drop_mask).to(torch.float)
            final_edge_mask = final_edge_mask + edge_sel_mask * alpha
            sel_mask = ~drop_mask
            sel_mask = sel_mask.to(torch.float)
            final_mask = final_mask + sel_mask * alpha
        # edge_weights = edge_weights * final_edge_mask
        # edge_weights = torch.where(edge_weights < 0.6, zero_tensor, one_tensor)
        final_mask = final_mask.unsqueeze(1).expand(x.size())
        x = x.mul(final_mask)

    elif aug_ops is edge_minus:
        final_edge_mask = torch.zeros((edge_index.shape[1],), device=x.device)
        for i in range(len(aug_ops)):
            aug_op = aug_ops[i]
            alpha = alphas[i]
            if aug_op is None:
                drop_mask = torch.zeros(
                    (edge_index.shape[1],), dtype=torch.bool, device=x.device
                )
            elif aug_op in [
                edge_dropping_1,
                edge_dropping_2,
                edge_dropping_3,
                edge_dropping_4,
                edge_dropping_5,
            ]:
                drop_mask = aug_op(edge_index)
            elif aug_op in [edge_dropping_6, edge_dropping_7]:
                bi_index, bi_weights = to_undirected(
                    edge_index, edge_weights, num_nodes=N
                )
                bi_index, bi_weights = add_self_loops(bi_index, bi_weights, num_nodes=N)
                deg = scatter(bi_weights, bi_index[0])
                drop_mask = aug_op(deg, edge_index)
            elif aug_op in [edge_dropping_8, edge_dropping_9]:
                drop_mask = aug_op(lg_s)
            else:
                drop_mask = aug_op(edge_index, N, pred, conf, T, epoch, t)
            edge_sel_mask = (~drop_mask).to(torch.float)
            final_edge_mask = final_edge_mask + edge_sel_mask * alpha
        edge_weights = edge_weights * final_edge_mask
        if mode != "evaluate_single_path":
            edge_weights = (edge_weights - edge_weights.min()) / (
                edge_weights.max() - edge_weights.min()
            )
            q = torch.tensor([0.1], device=edge_weights.device)
            T = edge_weights.quantile(q)
            edge_weights = torch.sigmoid(
                (edge_weights - T) * 1e5
            )  # approximate to 0 or 1
            # edge_weights = 1e5 * (edge_weights * final_edge_mask - 0.7)   not very good
            # edge_weights = torch.where(edge_weights < 0.7, zero_tensor, one_tensor) nondifferentiable
            # sigmoid = nn.Sigmoid()
            # edge_weights = sigmoid(edge_weights)

    elif aug_ops is attr_perturb:
        final_mask = torch.zeros_like(x, device=x.device)
        for i in range(len(aug_ops)):
            aug_op = aug_ops[i]
            alpha = alphas[i]
            if aug_op is None:
                drop_mask = torch.zeros_like(
                    torch.abs(x), dtype=torch.bool, device=x.device
                )
            elif aug_op in [attr_perturbing_1, attr_perturbing_2, attr_perturbing_3]:
                drop_mask = aug_op(torch.abs(x))
            elif aug_op in [attr_perturbing_4, attr_perturbing_5]:
                bi_index, bi_weights = to_undirected(
                    edge_index, edge_weights, num_nodes=N
                )
                bi_index, bi_weights = add_self_loops(bi_index, bi_weights, num_nodes=N)
                deg = scatter(bi_weights, bi_index[0])
                drop_mask = aug_op(torch.abs(x), deg)
            else:
                drop_mask = aug_op(torch.abs(x), node_s)
            sel_mask = ~drop_mask
            sel_mask = sel_mask.to(torch.float)
            final_mask = final_mask + sel_mask * alpha
        x = x.mul(final_mask)

    else:
        final_x = torch.zeros_like(x, device=x.device)
        for i in range(len(aug_ops)):
            aug_op = aug_ops[i]
            alpha = alphas[i]
            if aug_op is None:
                final_x = final_x + x * alpha
            elif aug_op in [attr_adding_1, attr_adding_2]:
                new_x = aug_op(x, edge_index, edge_weights)
                final_x = final_x + new_x * alpha
            elif aug_op in [attr_adding_7, attr_adding_8]:
                new_x = aug_op(x, edge_index, edge_weights)
                final_x = final_x + new_x * alpha
            else:
                new_x = aug_op(x)
                final_x = final_x + new_x * alpha
        x = final_x

    return edge_weights, x


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average="micro")
    return micro


def consis_loss(args, logps):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.0
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)

    sharp_p = (
        torch.pow(avg_p, 1.0 / args.tem)
        / torch.sum(torch.pow(avg_p, 1.0 / args.tem), dim=1, keepdim=True)
    ).detach()

    loss = 0.0
    conf_indices = avg_p.max(1)[0] > args.conf
    for p in ps:
        if args.reg_loss == "kl":
            loss += torch.mean((-sharp_p * torch.log(p)).sum(1)[conf_indices])
        elif args.reg_loss == "l2":
            loss += torch.mean((p - sharp_p).pow(2).sum(1)[conf_indices])
        else:
            raise ValueError(f"Unknown loss type: {args.loss}")
    loss = loss / len(ps)
    return loss, avg_p


def clip_grad_norm(params, max_norm):
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience,
        save_model=False,
        save_alphas=False,
        verbose=False,
        delta=0.0,
        save_model_path="checkpoint.pt",
        save_alphas_path="alphas.pkl",
        use_loss=True,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.save_model = save_model
        self.save_alphas = save_alphas
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = 0.0
        self.delta = delta
        self.save_model_path = "./checkpoint/" + save_model_path
        self.save_alphas_path = "./checkpoint/" + save_alphas_path
        self.use_loss = use_loss

    def __call__(self, val, model, current_epoch=-1):
        if self.use_loss:
            score = -val
        else:
            score = val

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = current_epoch
            if self.save_model:
                self.save_checkpoint(val, model)
            if self.save_alphas:
                self.save_variables(model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = current_epoch
            if self.save_model:
                self.save_checkpoint(val, model)
            if self.save_alphas:
                self.save_variables(model)
            self.counter = 0

    def save_checkpoint(self, val, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            if self.use_loss:
                print(
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val:.6f}).  Saving model ..."
                )
            else:
                print(
                    f"Validation acc increased ({self.val_acc_max:.6f} --> {val:.6f}).  Saving model ..."
                )
        torch.save(model.state_dict(), self.save_model_path)
        if self.use_loss:
            self.val_loss_min = val
        else:
            self.val_acc_max = val

    def save_variables(self, model):
        alphas = [alpha.data for alpha in model.arch_parameters()]
        pkl.dump(alphas, open(self.save_alphas_path, "wb"))
