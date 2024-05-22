from os import replace
import numpy as np
import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_add
import dgl
import scipy.sparse as sp
from utils import *


def get_sim(embeds1, embeds2):
    # normalize embeddings across feature dimension
    embeds1 = F.normalize(embeds1)
    embeds2 = F.normalize(embeds2)
    sim = th.mm(embeds1, embeds2.t())
    return sim


def neighbor_pruning(graph, node_dist, sim, x, degree, feat_drop_rate_1, feat_drop_rate_2, threshold):

    feat1 = drop_feature(x, feat_drop_rate_1)
    feat2 = drop_feature(x, feat_drop_rate_2)

    max_degree = np.max(degree)

    src_idx = th.LongTensor(np.argwhere(degree < threshold).flatten())
    rest_idx = th.LongTensor(np.argwhere(degree >= threshold).flatten())
    
    node_degree = degree[(degree<=threshold) & (degree > 2)]
    rest_node_degree = degree[degree>=threshold]

    sim = get_sim(x, x)
    sim = th.clamp(sim, 0, 1)
    sim = sim - th.diag_embed(th.diag(sim))

    node_degree = th.LongTensor(node_degree)
    rest_node_degree = th.LongTensor(rest_node_degree)

    degree_dist = scatter_add(th.ones(node_degree.size()), node_degree)

    
    prob = degree_dist.unsqueeze(dim=0).repeat(src_idx.size(0), 1)
    
    aug_degree = th.multinomial(prob, 1).flatten()
    
    src_nodes, dst_nodes = graph.edges()
    src_edge_indices = np.where(np.isin(src_nodes.numpy(), src_idx))[0]
    dst_edge_indices = np.where(np.isin(dst_nodes.numpy(), src_idx))[0]

    combined_indices = np.unique(np.concatenate((src_edge_indices, dst_edge_indices)))
    new_row_mix, new_col_mix = src_nodes[combined_indices], dst_nodes[combined_indices]

    new_row_rest, new_col_rest = degree_mask_edge_threshold(node_dist, rest_idx, sim, max_degree, rest_node_degree, threshold)

    nsrc = np.concatenate((new_row_mix, new_row_rest))
    ndst = np.concatenate((new_col_mix, new_col_rest.cpu()))
    
    ng = dgl.graph((nsrc, ndst), num_nodes=graph.number_of_nodes())
    ng = dgl.transform.remove_self_loop(ng)

    ng = dgl.transform.add_self_loop(ng)

    adj = ng.adjacency_matrix().to_dense()
    adj = sp.csr_matrix(adj)
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj = torch.FloatTensor(adj[np.newaxis])

    return ng, feat1.unsqueeze(0), feat2.unsqueeze(0), adj


def neighbor_completion(graph, node_dist, sim, ano_sim, x, degree, 
        feat_drop_rate_1, edge_mask_rate_1, feat_drop_rate_2, edge_mask_rate_2, 
        threshold, device):

    feat1 = drop_feature(x, feat_drop_rate_1)
    feat2 = drop_feature(x, feat_drop_rate_2)

    max_degree = np.max(degree)
    
    src_idx = th.LongTensor(np.argwhere(degree < threshold).flatten())
    rest_idx = th.LongTensor(np.argwhere(degree >= threshold).flatten())
    
    node_degree = degree[(degree<=threshold) & (degree > 2)]
    rest_node_degree = degree[degree>=threshold]
    
    sim = get_sim(x, x)
    sim = th.clamp(sim, 0, 1)
    sim = sim - th.diag_embed(th.diag(sim))
    
    sim = sim * ano_sim
    sim = th.clamp(sim, 0, 1)
    src_sim = sim[src_idx]
   
    dst_idx = th.multinomial(src_sim + 1e-12, 1).flatten()

    dst_idx_list = []
  
    node_degree = th.LongTensor(node_degree)
    rest_node_degree = th.LongTensor(rest_node_degree)
   
    degree_dist = scatter_add(th.ones(node_degree.size()), node_degree)

    
    prob = degree_dist.unsqueeze(dim=0).repeat(src_idx.size(0), 1)
    aug_degree = th.multinomial(prob, 1).flatten()
  
    new_row_mix_1, new_col_mix_1 = mixup(src_idx, dst_idx, node_dist, sim, max_degree, aug_degree, device)
    new_row_rest_1, new_col_rest_1 = degree_mask_edge(node_dist, rest_idx, sim, max_degree, rest_node_degree, edge_mask_rate_1)

    nsrc1 = th.cat((new_row_mix_1, new_row_rest_1)).cpu()
    ndst1 = th.cat((new_col_mix_1, new_col_rest_1)).cpu()

    ng1 = dgl.graph((nsrc1, ndst1), num_nodes=graph.number_of_nodes())
    ng1 = dgl.transform.remove_self_loop(ng1)
    
    ng1 = dgl.transform.add_self_loop(ng1)

    new_row_mix_2, new_col_mix_2 = mixup(src_idx, dst_idx, node_dist, sim, max_degree, aug_degree, device)
    new_row_rest_2, new_col_rest_2 = degree_mask_edge(node_dist, rest_idx, sim, max_degree, rest_node_degree, edge_mask_rate_2)

    nsrc2 = th.cat((new_row_mix_2, new_row_rest_2)).cpu()
    ndst2 = th.cat((new_col_mix_2, new_col_rest_2)).cpu()

    ng2 = dgl.graph((nsrc2, ndst2), num_nodes=graph.number_of_nodes())
    ng2 = dgl.transform.remove_self_loop(ng2)

    ng2 = dgl.transform.add_self_loop(ng2)

    adj1 = ng1.adjacency_matrix().to_dense()
    adj1 = sp.csr_matrix(adj1)
    adj1 = normalize_adj(adj1)
    adj1 = (adj1 + sp.eye(adj1.shape[0])).todense()
    adj1 = torch.FloatTensor(adj1[np.newaxis])

    adj2 = ng2.adjacency_matrix().to_dense()
    adj2 = sp.csr_matrix(adj2)
    adj2 = normalize_adj(adj2)
    adj2 = (adj2 + sp.eye(adj2.shape[0])).todense()
    adj2 = torch.FloatTensor(adj2[np.newaxis])

    return ng1, ng2, feat1.unsqueeze(0), feat2.unsqueeze(0), adj1, adj2


def drop_feature(x, drop_prob):
    drop_mask = th.empty((x.size(1),),
                    dtype=th.float32).uniform_(0, 1) < drop_prob

    x = x.clone()
    x[:, drop_mask] = 0
    return x


def mixup(src_idx, dst_idx, node_dist, sim, 
                    max_degree, aug_degree, device):
    phi = sim[src_idx, dst_idx].unsqueeze(dim=1).to(device)
    phi = th.clamp(phi, 0, 0.5)

    mix_dist = sim[dst_idx] * node_dist[dst_idx] * phi + sim[src_idx] * node_dist[src_idx] * (1-phi)

    new_tgt = th.multinomial(mix_dist + 1e-12, int(max_degree))
    
    tgt_idx = th.arange(max_degree).unsqueeze(dim=0)
    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]

    new_row = src_idx.repeat_interleave(aug_degree)

    return new_row, new_col
    

def degree_mask_edge(adj, idx, sim, max_degree, node_degree, mask_prob):
    aug_degree = (node_degree * (1- mask_prob)).long()
    sim_dist = sim[idx] * adj[idx]

    new_tgt = th.multinomial(sim_dist + 1e-12, int(max_degree))
    tgt_idx = th.arange(max_degree).unsqueeze(dim=0)

    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = idx.repeat_interleave(aug_degree)
    
    return new_row, new_col

def degree_mask_edge_threshold(adj, idx, sim, max_degree, node_degree, threshold):
    aug_degree = th.full_like(node_degree, fill_value=threshold)

    sim_dist = sim[idx] * adj[idx]

    new_tgt = th.multinomial(sim_dist + 1e-12, int(max_degree))
    tgt_idx = th.arange(max_degree).unsqueeze(dim=0)

    new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = idx.repeat_interleave(aug_degree)

    return new_row, new_col