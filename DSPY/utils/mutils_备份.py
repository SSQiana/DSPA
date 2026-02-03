import random
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import shutil
from torch.utils.tensorboard import SummaryWriter
import json
import os.path as osp


def get_arg_dict(args):
    info_dict = args.__dict__
    ks = list(info_dict.keys())
    arg_dict = {}
    for k in ks:
        v = info_dict[k]
        for t in [int, float, str, bool, torch.Tensor]:
            if isinstance(v, t):
                arg_dict[k] = v
                break
    return arg_dict


class EarlyStopping:
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def reset(self):
        self.best = None

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def is_empty_edges(edges):
    return edges.shape[1] == 0


def map2id(l):
    return dict(zip(l, range(len(l))))


def sorteddict(x, min=True, dim=1):
    if min:
        return dict(sorted(x.items(), key=lambda item: item[dim]))
    else:
        return dict(sorted(x.items(), key=lambda item: item[dim])[::-1])


# from gensim.models import Word2Vec
# import gensim
# def sen2vec(sentences, vector_size=32):
#     """use gensim.word2vec to generate wordvecs, and average as sentence vectors.
#     if exception happens use zero embedding.
#     @ params sentences : list of sentence
#     @ params vector_size
#     @ return : sentence embedding
#     """
#     sentences = [list(gensim.utils.tokenize(a, lower=True)) for a in sentences]
#     model = Word2Vec(sentences, vector_size=vector_size, min_count=1)
#     print('word2vec done')
#     embs = []
#     for s in sentences:
#         try:
#             emb = model.wv[s]
#             emb = np.mean(emb, axis=0)
#         except Exception as e:
#             print(e)
#             emb = np.zeros(vector_size)
#         embs.append(emb)
#     embs = np.stack(embs)
#     print(f'emb shape : {embs.shape}')
#     return embs


# from torch_geometric.utils.negative_sampling import negative_sampling
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.utils import coalesce, degree, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


def negative_sampling(edge_index: Tensor,
                      num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                      num_neg_samples: Optional[int] = None,
                      method: str = "sparse",
                      force_undirected: bool = False) -> Tensor:
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return.
            If set to :obj:`None`, will try to return a negative edge for every
            positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:

        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> negative_sampling(edge_index)
        tensor([[3, 0, 0, 3],
                [2, 3, 2, 1]])

        >>> # For bipartite graph
        >>> negative_sampling(edge_index, num_nodes=(3, 4))
        tensor([[0, 2, 2, 1],
                [2, 2, 1, 3]])
    """
    assert method in ['sparse', 'dense']

    size = num_nodes
    bipartite = isinstance(size, (tuple, list))
    size = maybe_num_nodes(edge_index) if size is None else size
    size = (size, size) if not bipartite else size
    force_undirected = False if bipartite else force_undirected

    idx, population = edge_index_to_vector(edge_index, size, bipartite,
                                           force_undirected)

    if idx.numel() >= population:
        return edge_index.new_empty((2, 0))

    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
    if force_undirected:
        num_neg_samples = num_neg_samples // 2

    prob = 1. - idx.numel() / population  # Probability to sample a negative.
    sample_size = int(1.1 * num_neg_samples / prob)  # (Over)-sample size.

    neg_idx = None
    if method == 'dense':
        # The dense version creates a mask of shape `population` to check for
        # invalid samples.
        mask = idx.new_ones(population, dtype=torch.bool)
        mask[idx] = False
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, idx.device)
            rnd = rnd[mask[rnd]]  # Filter true negatives.
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            mask[neg_idx] = False

    else:  # 'sparse'
        # The sparse version checks for invalid samples via `np.isin`.
        idx = idx.to('cpu')
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, device='cpu')
            mask = np.isin(rnd, idx)
            if neg_idx is not None:
                mask |= np.isin(rnd, neg_idx.to('cpu'))
            mask = torch.from_numpy(mask).to(torch.bool)
            rnd = rnd[~mask].to(edge_index.device)
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break

    return vector_to_edge_index(neg_idx, size, bipartite, force_undirected)


def batched_negative_sampling(
    edge_index: Tensor,
    batch: Union[Tensor, Tuple[Tensor, Tensor]],
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of multiple graphs given by
    :attr:`edge_index` and :attr:`batch`.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor or Tuple[LongTensor, LongTensor]): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph connecting two different node types.
        num_neg_samples (int, optional): The number of negative samples to
            return. If set to :obj:`None`, will try to return a negative edge
            for every positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:

        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
        >>> edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
        >>> edge_index
        tensor([[0, 0, 1, 2, 4, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> batched_negative_sampling(edge_index, batch)
        tensor([[3, 1, 3, 2, 7, 7, 6, 5],
                [2, 0, 1, 1, 5, 6, 4, 4]])

        >>> # For bipartite graph
        >>> edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
        >>> edge_index2 = edge_index1 + torch.tensor([[2], [4]])
        >>> edge_index3 = edge_index2 + torch.tensor([[2], [4]])
        >>> edge_index = torch.cat([edge_index1, edge_index2,
        ...                         edge_index3], dim=1)
        >>> edge_index
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
        >>> src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        >>> batched_negative_sampling(edge_index,
        ...                           (src_batch, dst_batch))
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9]])
    """
    if isinstance(batch, Tensor):
        src_batch, dst_batch = batch, batch
    else:
        src_batch, dst_batch = batch[0], batch[1]

    split = degree(src_batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)

    num_src = degree(src_batch, dtype=torch.long)
    cum_src = torch.cat([src_batch.new_zeros(1), num_src.cumsum(0)[:-1]])

    if isinstance(batch, Tensor):
        num_nodes = num_src.tolist()
        cumsum = cum_src
    else:
        num_dst = degree(dst_batch, dtype=torch.long)
        cum_dst = torch.cat([dst_batch.new_zeros(1), num_dst.cumsum(0)[:-1]])

        num_nodes = torch.stack([num_src, num_dst], dim=1).tolist()
        cumsum = torch.stack([cum_src, cum_dst], dim=1).unsqueeze(-1)

    neg_edge_indices = []
    for i, edge_index in enumerate(edge_indices):
        edge_index = edge_index - cumsum[i]
        neg_edge_index = negative_sampling(edge_index, num_nodes[i],
                                           num_neg_samples, method,
                                           force_undirected)
        neg_edge_index += cumsum[i]
        neg_edge_indices.append(neg_edge_index)

    return torch.cat(neg_edge_indices, dim=1)


def structured_negative_sampling(edge_index, num_nodes: Optional[int] = None,
                                 contains_neg_self_loops: bool = True):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: (LongTensor, LongTensor, LongTensor)

    Example:

        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> structured_negative_sampling(edge_index)
        (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)

    rand = torch.randint(num_nodes, (row.size(0), ), dtype=torch.long)
    neg_idx = row * num_nodes + rand

    mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.size(0), ), dtype=torch.long)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)


def structured_negative_sampling_feasible(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> bool:
    r"""Returns :obj:`True` if
    :meth:`~torch_geometric.utils.structured_negative_sampling` is feasible
    on the graph given by :obj:`edge_index`.
    :meth:`~torch_geometric.utils.structured_negative_sampling` is infeasible
    if atleast one node is connected to all other nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: bool

    Examples:

        >>> edge_index = torch.LongTensor([[0, 0, 1, 1, 2, 2, 2],
        ...                                [1, 2, 0, 2, 0, 1, 1]])
        >>> structured_negative_sampling_feasible(edge_index, 3, False)
        False

        >>> structured_negative_sampling_feasible(edge_index, 3, True)
        True
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    max_num_neighbors = num_nodes

    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    if not contains_neg_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
        max_num_neighbors -= 1  # Reduce number of valid neighbors

    deg = degree(edge_index[0], num_nodes)
    # True if there exists no node that is connected to all other nodes.
    return bool(torch.all(deg < max_num_neighbors))


###############################################################################


def sample(population: int, k: int, device=None) -> Tensor:
    if population <= k:
        return torch.arange(population, device=device)
    else:
        return torch.tensor(random.sample(range(population), k), device=device)


def edge_index_to_vector(
    edge_index: Tensor,
    size: Tuple[int, int],
    bipartite: bool,
    force_undirected: bool = False,
) -> Tuple[Tensor, int]:

    row, col = edge_index

    if bipartite:  # No need to account for self-loops.
        idx = (row * size[1]).add_(col)
        population = size[0] * size[1]
        return idx, population

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We only operate on the upper triangular matrix:
        mask = row < col
        row, col = row[mask], col[mask]
        offset = torch.arange(1, num_nodes, device=row.device).cumsum(0)[row]
        idx = row.mul_(num_nodes).add_(col).sub_(offset)
        population = (num_nodes * (num_nodes + 1)) // 2 - num_nodes
        return idx, population

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We remove self-loops as we do not want to take them into account
        # when sampling negative values.
        mask = row != col
        row, col = row[mask], col[mask]
        col[row < col] -= 1
        idx = row.mul_(num_nodes - 1).add_(col)
        population = num_nodes * num_nodes - num_nodes
        return idx, population


def vector_to_edge_index(idx: Tensor, size: Tuple[int, int], bipartite: bool,
                         force_undirected: bool = False) -> Tensor:

    if bipartite:  # No need to account for self-loops.
        row = idx.div(size[1], rounding_mode='floor')
        col = idx % size[1]
        return torch.stack([row, col], dim=0)

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        offset = torch.arange(1, num_nodes, device=idx.device).cumsum(0)
        end = torch.arange(num_nodes, num_nodes * num_nodes, num_nodes,
                           device=idx.device)
        row = torch.bucketize(idx, end.sub_(offset), right=True)
        col = offset[row].add_(idx) % num_nodes
        return torch.stack([torch.cat([row, col]), torch.cat([col, row])], 0)

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        row = idx.div(num_nodes - 1, rounding_mode='floor')
        col = idx % (num_nodes - 1)
        col[row <= col] += 1
        return torch.stack([row, col], dim=0)


def hard_negative_sampling(edges, all_neg=False, inplace=False):
    ei = edges.numpy()

    # reorder
    nodes = list(set(ei.flatten()))
    nodes.sort()
    id2n = nodes
    n2id = dict(zip(nodes, np.arange(len(nodes))))

    ei_ = np.vectorize(lambda x: n2id[x])(ei)

    if all_neg:
        maxn = len(nodes)
        nei_ = []
        pos_e = set([tuple(x) for x in ei_.T])
        for i in range(maxn):
            for j in range(maxn):
                if i != j and (i, j) not in pos_e:
                    nei_.append([i, j])
        nei_ = torch.LongTensor(nei_).T
    else:
        nei_ = negative_sampling(torch.LongTensor(ei_))
    nei = torch.LongTensor(np.vectorize(lambda x: id2n[x])(nei_.numpy()))
    return nei


def bi_negative_sampling(edges, num_nodes, shift):
    nes = edges.new_zeros((2, 0))
    while nes.shape[1] < edges.shape[1]:
        num_need = edges.shape[1] - nes.shape[1]
        ne = negative_sampling(edges, num_nodes=num_nodes, num_neg_samples=4 * num_need)
        mask = torch.logical_xor((ne[0] < shift), (ne[1] < shift))
        ne = ne[:, mask]
        nes = torch.cat([nes, ne[:, :num_need]], dim=-1)
    return nes


import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))



