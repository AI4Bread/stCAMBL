import os
import torch
import numpy as np
import scanpy as sc
import torch.nn as nn

def isin(tensor, values):
    values = values.to(tensor.device)
    return torch.any(torch.eq(tensor.unsqueeze(1), values), dim=1)

def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

def drop_edges(adj, drop_prob=0.5):
    if not isinstance(adj, torch.sparse.Tensor):
        raise ValueError("Input must be a sparse tensor.")
    adj_coo = adj.coalesce()
    indices = adj_coo.indices()
    values = adj_coo.values()
    edge_mask = torch.rand(indices.size(1), device=indices.device) > drop_prob
    new_indices = indices[:, edge_mask]
    new_values = values[edge_mask]
    new_adj = torch.sparse_coo_tensor(new_indices, new_values, adj.shape, device=adj.device)
    new_adj = new_adj.coalesce()  

    return new_adj


# def extract_values_from_adj(adj, edge_index):
#     adj_coo = adj.coalesce()
#     adj_indices = adj_coo.indices()
#     adj_values = adj_coo.values()

#     edge_coo = torch.stack([edge_index[0], edge_index[1]])

#     mask = torch.zeros(adj_indices.size(1), dtype=torch.bool, device=adj.device)
#     for i in range(edge_coo.size(1)):
#         row = edge_coo[0, i]
#         col = edge_coo[1, i]
#         matches1 = (adj_indices[0] == row) & (adj_indices[1] == col)
#         matches2 = (adj_indices[0] == col) & (adj_indices[1] == row)
#         matches = matches1 | matches2
#         mask = mask.logical_or(matches)
#     extracted_values = adj_values[mask]

#     if extracted_values.size(0) != edge_coo.size(1):
#         raise ValueError(f"Extracted values count {extracted_values.size(0)} does not match edge_index count {edge_coo.size(1)}")

#     return extracted_values

# def edge_to_sparse_adj(n, edge, values=None):

#     indices = torch.stack([edge[0], edge[1]])
#     if values is None:
#         values = torch.ones(indices.size(1), dtype=torch.float, device=indices.device)
#     sparse_adj_matrix = torch.sparse_coo_tensor(indices, values, (n, n), device=indices.device)
#     sparse_adj_matrix = sparse_adj_matrix.coalesce()
#     return sparse_adj_matrix

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

class WeightInit(nn.Module):
    def __init__(self, n_h):
        super(WeightInit, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


def set_seed(seed):
    import random
    import torch
    from torch.backends import cudnn

    #seed = 666
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
