###
# Modified based on SPAN: https://github.com/Louise-LuLin/GCL-SPAN
###

from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List

from tqdm import tqdm
import pickle as pkl
import os
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor

from torch_geometric.utils.sparse import to_edge_index
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_geometric.data import Batch, Data
from utils import get_adj_tensor, to_dense_adj, dense_to_sparse, switch_edge, drop_feature
import time
from torch_geometric.utils import degree, subgraph, to_undirected

eps = 1e-5


###################### Base Class ######################

class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    ptb_prob: Optional[SparseTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[SparseTensor]]:
        return self.x, self.edge_index, self.ptb_prob


class Augmentor(ABC):
    """Base class for graph augmentors."""

    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self,
            x: torch.FloatTensor,
            edge_index: torch.LongTensor,
            ptb_prob: Optional[SparseTensor] = None,
            batch=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.augment(Graph(x, edge_index, ptb_prob), batch).unfold()


# def get_normalize_adj_tensor(adj, device):
#     # device = 'cuda:0'
#     # device = torch.device(device if adj.is_cuda else "cpu")
#
#     mx = adj + torch.eye(adj.shape[0]).to(device)
#     rowsum = mx.sum(1)
#     r_inv = rowsum.pow(-1 / 2).flatten()
#     r_inv[torch.isinf(r_inv)] = 0.
#     r_mat_inv = torch.diag(r_inv)
#     mx = r_mat_inv @ mx
#     mx = mx @ r_mat_inv
#
#     return mx



# def get_normalize_adj_tensor(adj, device='cuda:0'):
#     if not adj.is_sparse:
#         adj = adj.to_sparse_coo()
#     # Infer the device from the input tensor
#     device = adj.device
#     num_nodes = adj.shape[0]
#     indices_I = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
#     values_I = torch.ones(num_nodes, device=device, dtype=torch.float)
#     I = torch.sparse_coo_tensor(indices_I, values_I, adj.shape)
#     mx = (adj + I).coalesce()
#     rowsum = torch.sparse.sum(mx, dim=1).to_dense()
#     r_inv_sqrt = torch.pow(rowsum, -0.5)
#     r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
#     indices = mx.indices()
#     values = mx.values()
#     row, col = indices[0], indices[1]
#     norm_values = values * r_inv_sqrt[row] * r_inv_sqrt[col]
#     normalized_adj = torch.sparse_coo_tensor(indices, norm_values, mx.shape)
#     return normalized_adj.to_dense()

def get_normalize_adj_tensor(adj, device='cuda:0'):
    # 1. 转换稀疏格式 (保持原逻辑)
    if not adj.is_sparse:
        adj = adj.to_sparse_coo()

    device = adj.device
    num_nodes = adj.shape[0]

    # 2. 添加自环 (A + I)
    indices_I = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
    values_I = torch.ones(num_nodes, device=device, dtype=torch.float)
    I = torch.sparse_coo_tensor(indices_I, values_I, adj.shape)

    mx = (adj + I).coalesce()

    rowsum = torch.sparse.sum(mx, dim=1).to_dense()

    r_inv_sqrt = torch.pow(rowsum + 1e-12, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.

    indices = mx.indices()
    values = mx.values()
    row, col = indices[0], indices[1]

    norm_values = values * r_inv_sqrt[row] * r_inv_sqrt[col]
    normalized_adj_sparse = torch.sparse_coo_tensor(indices, norm_values, mx.shape)

    normalized_adj_dense = normalized_adj_sparse.to_dense()

    if torch.isnan(normalized_adj_dense).any():
        normalized_adj_dense = torch.nan_to_num(normalized_adj_dense, nan=0.0)
    normalized_adj_dense = (normalized_adj_dense + normalized_adj_dense.t()) / 2.0

    return normalized_adj_dense





###################### Customized Class ######################

# compose multiple augmentors
class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g, batch)
        return g


# feature augmentor
class FeatureAugmentor(Augmentor):
    def __init__(self, pf: float):
        super(FeatureAugmentor, self).__init__()
        self.pf = pf

    def augment(self, g: Graph, batch: torch.Tensor) -> Graph:
        x, edge_index, _ = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, ptb_prob=None)

    def get_aug_name(self):
        return 'feature'


# spectral augmentorclass
class SpectralAugmentor(Augmentor):
    def __init__(self, ratio, lr, iteration, dis_type, device, sample='no', threshold=0.5, k=5):
        super(SpectralAugmentor, self).__init__()
        self.ratio = ratio
        self.lr = lr
        self.iteration = iteration
        self.dis_type = dis_type
        self.device = device
        self.sample = sample
        self.threshold = threshold
        self.k_perturbations = k

    def get_aug_name(self):
        return self.dis_type

    def calc_prob(self, args, data, ix, fast=True, check='no', save='no', verbose=False, silence=False):
        debug_check = True

        mode = 'max'
        if args.dataset == 'aminer':
            edge_index = data['edge_index'][ix]
            total_num_nodes = data['x'].shape[0]
        elif args.dataset in ['collab_04', 'collab_06', 'collab_08']:
            edge_index = data['train']['edge_index_list'][ix]
            total_num_nodes = data['x'].shape[1]
        else:
            edge_index = data['train']['edge_index_list'][ix]
            total_num_nodes = data['x'].shape[0]

        device = self.device
        edge_index = edge_index.to(device)

        deg = degree(edge_index[0], total_num_nodes, dtype=torch.long) + \
              degree(edge_index[1], total_num_nodes, dtype=torch.long)

        # Mask for active nodes
        active_mask = (deg > 0)
        active_nodes = torch.where(active_mask)[0]
        sub_num_nodes = active_nodes.size(0)

        if sub_num_nodes == 0:
            empty_adj = SparseTensor(
                row=torch.tensor([], dtype=torch.long),
                col=torch.tensor([], dtype=torch.long),
                value=torch.tensor([]),
                sparse_sizes=(total_num_nodes, total_num_nodes)
            ).to(device)

            if hasattr(self, 'dis_type'):
                data[self.dis_type] = empty_adj

            return data

        if sub_num_nodes < 2:
            sym_edge_index = to_undirected(edge_index)
            adj_t = SparseTensor(
                row=sym_edge_index[0],
                col=sym_edge_index[1],
                sparse_sizes=(total_num_nodes, total_num_nodes)
            ).to(device)

            if hasattr(self, 'dis_type'):
                data[self.dis_type] = adj_t

            return data

        sub_edge_index, _ = subgraph(active_nodes, edge_index, relabel_nodes=True)

        ori_adj = get_adj_tensor(sub_edge_index).to(device)  # Dense Matrix

        ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=device)

        matrix_on_cpu = ori_adj_norm.cpu()
        if torch.isnan(matrix_on_cpu).any() or torch.isinf(matrix_on_cpu).any():
            # print(f"[Warning] Batch {ix}: Found NaN/Inf in adjacency matrix. Cleaning...")
            matrix_on_cpu = torch.nan_to_num(matrix_on_cpu, nan=0.0, posinf=1.0, neginf=0.0)

        matrix_on_cpu = (matrix_on_cpu + matrix_on_cpu.t()) / 2.0
        jitter = 1e-6 * torch.eye(matrix_on_cpu.size(0))
        matrix_on_cpu = matrix_on_cpu + jitter

        try:
            ori_e, ori_v = torch.linalg.eigh(ori_adj_norm)
        except Exception as e:
            matrix_on_cpu = ori_adj_norm.cpu()
            ori_e, ori_v = torch.linalg.eigh(matrix_on_cpu)
            ori_e = ori_e.to(device)
            ori_v = ori_v.to(device)

        eigen_norm = torch.norm(ori_e)

        perturbed_adjs = [edge_index]

        n_perturbations = int(self.ratio * (ori_adj.sum() / 2))
        if n_perturbations < 1: n_perturbations = 1

        lr = self.lr
        iteration = self.iteration

        nnodes = sub_num_nodes
        num_params = int(nnodes * (nnodes - 1) / 2)

        adj_changes_list = []
        for _ in range(self.k_perturbations):
            adj_changes_k = Parameter(torch.FloatTensor(num_params), requires_grad=True).to(device)
            torch.nn.init.uniform_(adj_changes_k, 0, 1e-4)
            adj_changes_list.append(adj_changes_k)
        verb = max(1, int(self.iteration / 10))
        for t in range(1, iteration + 1):
            reg_losses = []

            for adj_changes in adj_changes_list:
                modified_adj = self.get_modified_adj(ori_adj, self.reshape_m(nnodes, adj_changes))
                adj_norm_noise = get_normalize_adj_tensor(modified_adj, device=device)

                if fast:
                    temp = torch.matmul(adj_norm_noise, ori_v)
                    e = (ori_v * temp).sum(dim=0)
                else:
                    e = torch.linalg.eigvalsh(adj_norm_noise)

                eigen_self = torch.norm(e)
                eigen_mse = torch.norm(ori_e - e)

                if mode == 'max':
                    reg_loss = eigen_mse / eigen_norm
                elif mode == 'min':
                    reg_loss = -eigen_mse / eigen_norm

                reg_losses.append(reg_loss)

            reg_losses_tensor = torch.stack(reg_losses)

            if len(reg_losses_tensor) > 1:
                variance_loss = torch.var(reg_losses_tensor)
            else:
                variance_loss = reg_losses_tensor.mean()

            self.loss = variance_loss
            adj_grads = torch.autograd.grad(self.loss, adj_changes_list)
            current_lr = lr / np.sqrt(t + 1)

            for k_idx, adj_changes in enumerate(adj_changes_list):
                adj_changes.data.add_(current_lr * adj_grads[k_idx])
                self.projection(n_perturbations, adj_changes)

        with torch.no_grad():
            for adj_changes in adj_changes_list:
                k_val = n_perturbations
                if k_val > num_params: k_val = num_params

                _, topk_indices = torch.topk(adj_changes, k_val)

                m_discrete = torch.zeros_like(adj_changes)
                m_discrete[topk_indices] = 1.0

                m_mat = self.reshape_m(nnodes, m_discrete)

                modified_final = self.get_modified_adj(ori_adj, m_mat)
                local_indices = modified_final.nonzero().t()
                if local_indices.size(0) > 0:
                    global_row = active_nodes[local_indices[0]]
                    global_col = active_nodes[local_indices[1]]
                    aug_edge_index = torch.stack([global_row, global_col], dim=0)
                else:
                    aug_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

                perturbed_adjs.append(aug_edge_index)

        if args.dataset == 'aminer':
            data['aug_edge_index_list'].append(perturbed_adjs)
        else:
            if 'aug_edge_index_list' not in data['train']:
                data['train']['aug_edge_index_list'] = []
            data['train']['aug_edge_index_list'].append(perturbed_adjs)

        return data

    def projection(self, n_perturbations, adj_changes):
        if torch.clamp(adj_changes, 0, self.threshold).sum() > n_perturbations:
            left = (adj_changes).min()
            right = adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, 1e-4, adj_changes)
            adj_changes.data.copy_(torch.clamp(adj_changes.data - miu, min=0, max=1))
        else:
            adj_changes.data.copy_(torch.clamp(adj_changes.data, min=0, max=1))

    def get_modified_adj(self, ori_adj, m):
        nnodes = ori_adj.shape[1]
        complementary = (torch.ones_like(ori_adj) - torch.eye(nnodes).to(self.device) - ori_adj) - ori_adj
        modified_adj = complementary * m + ori_adj
        return modified_adj

    def reshape_m(self, nnodes, adj_changes):
        m = torch.zeros((nnodes, nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=nnodes, col=nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes
        m = m + m.t()
        return m

    def bisection(self, a, b, n_perturbations, epsilon, adj_changes):
        def func(x):
            return torch.clamp(adj_changes - x, 0, self.threshold).sum() - n_perturbations

        miu = a
        for _ in range(50):
            miu = (a + b) / 2
            if (func(miu) == 0.0): break
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        return miu

    def augment(self, edge_index, ptb_prob):
        batch = None
        ori_adj = edge_index
        # ori_adj = to_dense_adj(edge_index, batch)
        row, col = (ptb_prob > 0).nonzero(as_tuple=True)
        val = ptb_prob[row, col]
        ptb_idx, ptb_w = to_edge_index(SparseTensor(row=row, col=col, value=val, sparse_sizes=ptb_prob.size()))
        num_nodes = ptb_prob.shape[0]
        ptb_m = to_dense_adj(ptb_idx, batch, ptb_w, max_num_nodes=num_nodes)
        # ptb_m = to_dense_adj(ptb_idx, batch, ptb_w)
        ptb_adj = self.random_sample(ptb_m)
        modified_adj = self.get_modified_adj(ori_adj, ptb_adj).detach()
        self.check_adj_tensor(modified_adj)
        edge_index, _ = dense_to_sparse(modified_adj)
        return modified_adj

    def add_random_noise(self, ori_adj):
        nnodes = ori_adj.shape[0]
        noise = 1e-4 * torch.rand(nnodes, nnodes).to(self.device)
        return (noise + torch.transpose(noise, 0, 1)) / 2.0 + ori_adj

    def random_sample(self, edge_prop):
        with torch.no_grad():
            s = edge_prop.cpu().detach().numpy()
            # s = (s + np.transpose(s))
            if self.sample == 'yes':
                binary = np.random.binomial(1, s)
                mask = np.random.binomial(1, 0.7, s.shape)
                sampled = np.multiply(binary, mask)
            else:
                sampled = np.random.binomial(1, s)
            return torch.FloatTensor(sampled).to(self.device)

    #############################################################
    # check intermediate results

    def check_hist(self, adj_changes):
        with torch.no_grad():
            s = adj_changes.cpu().detach().numpy()
            stat = {}
            stat['1.0'] = (s == 1.0).sum()
            stat['(1.0,0.8)'] = (s > 0.8).sum() - (s == 1.0).sum()
            stat['[0.8,0.6)'] = (s > 0.6).sum() - (s > 0.8).sum()
            stat['[0.6,0.4)'] = (s > 0.4).sum() - (s > 0.6).sum()
            stat['[0.4,0.2)'] = (s > 0.2).sum() - (s > 0.4).sum()
            stat['[0.2,0.0]'] = (s > 0.0).sum() - (s > 0.2).sum()
            stat['0.0'] = (s == 0.0).sum()
            print(stat)

    def check_adj_tensor(self, adj):
        """Check if the modified adjacency is unweighted, all-zero diagonal.
        """
        # assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1, "Max value should be 1!"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj[0].diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"

    def check_changes(self, ori_adj, adj_changes, y):
        nnodes = ori_adj.shape[0]

        m = torch.zeros((nnodes, nnodes))
        tril_indices = torch.tril_indices(row=nnodes, col=nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = adj_changes.cpu()
        m = m + m.t()
        idx = torch.nonzero(m).numpy()
        m = m.detach().numpy()
        degree = ori_adj.sum(dim=1).cpu().numpy()
        idx2 = torch.nonzero(ori_adj.cpu()).numpy()

        stat = {'intra': 0, 'inter': 0, 'degree': [], 'inter_add': 0, 'inter_rm': 0, 'intra_add': 0, 'intra_rm': 0,
                'degree_add': [], 'degree_rm': []}
        for i in tqdm(idx):
            d = degree[i[0]] + degree[i[1]]
            if ori_adj[i[0], i[1]] == 1:  # rm
                if y[i[0]] == y[i[1]]:  # intra
                    stat['intra_rm'] += m[i[0], i[1]]
                if y[i[0]] != y[i[1]]:  # inter
                    stat['inter_rm'] += m[i[0], i[1]]
                stat['degree_rm'].append(d / 2)
            if ori_adj[i[0], i[1]] == 0:  # add
                if y[i[0]] == y[i[1]]:  # intra
                    stat['intra_add'] += m[i[0], i[1]]
                if y[i[0]] != y[i[1]]:  # inter
                    stat['inter_add'] += m[i[0], i[1]]
                stat['degree_add'].append(d / 2)
        for i in tqdm(idx2):
            d = degree[i[0]] + degree[i[1]]
            if y[i[0]] == y[i[1]]:  # intra
                stat['intra'] += 1
            if y[i[0]] != y[i[1]]:  # inter
                stat['inter'] += 1
            stat['degree'].append(d / 2)

        stat['degree_rm'] = sum(stat['degree_rm']) / (len(stat['degree_rm']) + 0.1)
        stat['degree_add'] = sum(stat['degree_add']) / (len(stat['degree_add']) + 0.1)
        stat['degree'] = sum(stat['degree']) / (len(stat['degree']) + 0.1)

        print(stat)

    def augment_on_the_fly(self, g: Graph) -> Graph:
        x, edge_index, edge_prob = g.unfold()
        x = x.to(self.device)
        ori_adj = get_adj_tensor(edge_index.cpu()).to(self.device)
        # ori_adj = to_dense_adj(edge_index)

        nnodes = ori_adj.shape[0]

        adj_changes = Parameter(torch.FloatTensor(int(nnodes * (nnodes - 1) / 2)), requires_grad=True).to(self.device)
        torch.nn.init.uniform_(adj_changes, 0.0, 0.001)

        ori_adj_norm = get_normalize_adj_tensor(ori_adj, device=self.device)
        # ori_e = torch.linalg.eigvalsh(ori_adj_norm)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        eigen_norm = torch.norm(ori_e)

        # print(ori_adj.shape, ori_adj_norm.shape)
        # exit('')

        n_perturbations = int(self.ratio * (ori_adj.sum() / 2))
        with tqdm(total=self.iteration, desc='Spectral Augment') as pbar:
            for t in range(1, self.iteration + 1):
                modified_adj = self.get_modified_adj(ori_adj, self.reshape_m(nnodes, adj_changes))

                # add noise to make the graph asymmetric
                modified_adj_noise = modified_adj
                modified_adj_noise = self.add_random_noise(modified_adj)
                adj_norm_noise = get_normalize_adj_tensor(modified_adj_noise, device=self.device)
                # e = torch.linalg.eigvalsh(adj_norm_noise)
                e, v = torch.symeig(adj_norm_noise, eigenvectors=True)
                eigen_self = torch.norm(e)

                # spectral distance
                eigen_mse = torch.norm(ori_e - e)

                if self.dis_type == 'l2':
                    reg_loss = eigen_mse / eigen_norm
                elif self.dis_type == 'normDiv':
                    reg_loss = eigen_self / eigen_norm
                else:
                    exit(f'unknown distance metric: {self.dis_type}')

                self.loss = reg_loss

                adj_grad = torch.autograd.grad(self.loss, adj_changes)[0]

                lr = self.lr / np.sqrt(t + 1)
                adj_changes.data.add_(lr * adj_grad)

                before_p = torch.clamp(adj_changes, 0, 1).sum()
                self.projection(n_perturbations, adj_changes)
                after_p = torch.clamp(adj_changes, 0, 1).sum()

                pbar.set_postfix(
                    {'reg_loss': reg_loss.item(), 'eigen_mse': eigen_mse.item(), 'before_p': before_p.item(),
                     'after_p': after_p.item()})
                pbar.update()

        adj_changes = self.random_sample(adj_changes)

        modified_adj = self.get_modified_adj(ori_adj, self.reshape_m(nnodes, adj_changes)).detach()
        self.check_adj_tensor(modified_adj)

        edge_index, _ = dense_to_sparse(modified_adj)
        return Graph(x=x, edge_index=edge_index)



