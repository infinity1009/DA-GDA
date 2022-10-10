from search_space import *
from torch_scatter import scatter


def mask4edge(edge_index, drop_mask):
    sel_mask_1, sel_mask_2 = edge_index[0], edge_index[1]
    zero_tensor = torch.zeros((edge_index.shape[1],), device='cuda')
    one_tensor = torch.ones((edge_index.shape[1],), device='cuda')
    sel_mask_1 = torch.where(
        drop_mask[sel_mask_1], zero_tensor, one_tensor).to(torch.bool)
    sel_mask_2 = torch.where(
        drop_mask[sel_mask_2], zero_tensor, one_tensor).to(torch.bool)
    sel_mask = sel_mask_1 & sel_mask_2
    return sel_mask


def add_self_loops_mean(edge_index, edge_attr, N):
    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
                        reduce='mean')
    edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_attr


def take_augmentation(aug_ops, nx_graph, x, edge_weights, edge_index, alphas, ebc, deg, evc, edge_x, pred, conf, T, epoch, t, mode):
    N = x.shape[0]
    f = x.shape[1]
    zero_tensor = torch.zeros_like(
        edge_weights, dtype=torch.float, device='cuda')
    one_tensor = torch.ones_like(
        edge_weights, dtype=torch.float, device='cuda')
    if aug_ops is node_minus:
        if mode == 'evaluate_single_path':
            deg = get_degree(nx_graph, N)
            evc = get_eigenvector_centrality(nx_graph, N)
        final_mask = torch.zeros((x.shape[0], ), device=x.device)
        final_edge_mask = torch.zeros((edge_index.shape[1], ), device=x.device)
        for i in range(len(aug_ops)):
            aug_op = aug_ops[i]
            alpha = alphas[i]
            if aug_op is None:
                drop_mask = torch.zeros(
                    (x.shape[0], ), dtype=torch.bool, device=x.device)
            elif aug_op in [node_dropping_1, node_dropping_2, node_dropping_3]:
                drop_mask = aug_op(x.shape[0])
            elif aug_op in [node_dropping_4, node_dropping_5]:
                drop_mask = aug_op(deg)
            else:
                drop_mask = aug_op(evc)
            edge_sel_mask = mask4edge(edge_index, drop_mask).to(torch.float)
            final_edge_mask = final_edge_mask + edge_sel_mask * alpha
            sel_mask = ~drop_mask
            sel_mask = sel_mask.to(torch.float)
            final_mask = final_mask + sel_mask * alpha
        edge_weights = edge_weights * final_edge_mask
        edge_weights = torch.where(edge_weights < 0.6, zero_tensor, one_tensor)
        final_mask = final_mask.unsqueeze(1)
        final_mask = final_mask.expand(N, f)
        x = x.mul(final_mask)

    elif aug_ops is edge_minus:
        final_edge_mask = torch.zeros((edge_index.shape[1], ), device=x.device)
        for i in range(len(aug_ops)):
            aug_op = aug_ops[i]
            alpha = alphas[i]
            if aug_op is None:
                drop_mask = torch.zeros(
                    (edge_index.shape[1], ), dtype=torch.bool, device=x.device)
            elif aug_op in [edge_dropping_1, edge_dropping_2, edge_dropping_3, edge_dropping_4, edge_dropping_5]:
                drop_mask = aug_op(edge_index)
            elif aug_op in [edge_dropping_6, edge_dropping_7]:
                drop_mask = aug_op(ebc)
            elif aug_op in [edge_dropping_8, edge_dropping_9]:
                drop_mask = aug_op(edge_x)
            else:
                drop_mask = aug_op(
                    edge_index, x.shape[0], pred, conf, T, epoch, t)
            edge_sel_mask = ~drop_mask
            edge_sel_mask = edge_sel_mask.to(torch.float)
            final_edge_mask = final_edge_mask + edge_sel_mask * alpha
        edge_weights = edge_weights * final_edge_mask
        edge_weights = torch.where(edge_weights < 0.7, zero_tensor, one_tensor)

    elif aug_ops is attr_perturb:
        if mode == 'evaluate_single_path':
            deg = get_degree(nx_graph, N)
            evc = get_eigenvector_centrality(nx_graph, N)
        final_mask = torch.zeros_like(x, device=x.device)
        for i in range(len(aug_ops)):
            aug_op = aug_ops[i]
            alpha = alphas[i]
            if aug_op is None:
                drop_mask = torch.zeros_like(
                    x, dtype=torch.bool, device=x.device)
            elif aug_op in [attr_perturbing_1, attr_perturbing_2, attr_perturbing_3]:
                drop_mask = aug_op(x)
            elif aug_op in [attr_perturbing_4, attr_perturbing_5]:
                drop_mask = aug_op(x, deg)
            else:
                drop_mask = aug_op(x, evc)
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
    correct = correct.sum()
    return correct / len(labels)


def to_networkx(data, edge_attrs=None, to_undirected: bool = False, remove_self_loops: bool = False):
    '''
    adapted from PyG latest
    '''
    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    edge_attrs = edge_attrs or []

    values = {}
    for key, value in data(*(edge_attrs)):
        if torch.is_tensor(value):
            value = value if value.dim() <= 1 else value.squeeze(-1)
            values[key] = value.tolist()
        else:
            values[key] = value

    to_undirected = "upper" if to_undirected is True else to_undirected
    to_undirected_upper = True if to_undirected == "upper" else False
    to_undirected_lower = True if to_undirected == "lower" else False

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected_upper and u > v:
            continue
        elif to_undirected_lower and u < v:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    return G
